from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score
from utils import GrayscaleToRgb
from operator import itemgetter


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class Selector:
    """
    returns indices (along Batch dimension) of input
    """

    def __init__(self):
        pass

    def __call__(self, logits):
        raise NotImplementedError


class SoftmaxSelector(Selector):
    def __init__(self, min_kept=0, max_kept=0, threshold=0.8):
        super(Selector, self).__init__()
        self._min_kept = min_kept
        self._max_kept = max_kept
        self._threshold = threshold

    def __call__(self, logits):
        threshold = self._threshold
        probs = F.softmax(logits.detach().cpu(), dim=1)
        probs = np.max(probs.data.numpy(), axis=1)
        probs_sorted = np.sort(probs)[::-1]
        inds_sorted = probs.argsort()[::-1]
        if probs[inds_sorted[self._min_kept]] < threshold:
            threshold = probs[inds_sorted[self._min_kept]]
        if self._max_kept > 0 and probs[inds_sorted[self._max_kept]] > threshold:
            threshold = probs[inds_sorted[self._max_kept]]
        inds_selected = inds_sorted[np.where(probs_sorted > threshold)[0]]
        return inds_selected


class EntropySelector(Selector):
    def __init__(self, min_kept=0, max_kept=0, threshold=0.8):
        super(Selector, self).__init__()
        self._min_kept = min_kept
        self._max_kept = max_kept
        self._threshold = threshold

    def __call__(self, logits):
        threshold = self._threshold
        probs = F.softmax(logits.detach().cpu(), dim=1).data.numpy()
        entropy = -np.sum(probs * np.log(probs), axis=1)
        entropy_sorted = np.sort(entropy)
        inds_sorted = entropy.argsort()
        if entropy[inds_sorted[self._min_kept]] > threshold:
            threshold = entropy[inds_sorted[self._min_kept]]
        inds_selected = inds_sorted[np.where(entropy_sorted <= threshold)[0]]
        return inds_selected


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # return F.log_softmax(x, dim=1)
        return x


class BCNN(nn.Module):
    def __init__(self):
        super(BCNN, self).__init__()
        p = 0.2
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.drop1 = nn.Dropout2d(p)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.drop2 = nn.Dropout2d(p)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.drop3 = nn.Dropout(p)
        self.fc2 = nn.Linear(500, 10)
        self.fc3 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.drop1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.drop2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.drop3(self.fc1(x)))
        y = self.fc2(x)
        s = self.fc3(x)
        return y, s


class ImageFolderDataset(Dataset):
    def __init__(self, data_root, data_list, transform=None, return_line=False):
        self.root = data_root
        self.transform = transform

        if isinstance(data_list, list):
            self.data_list = data_list
        else:
            with open(data_list, 'r') as f:
                self.data_list = [l.strip('\n') for l in f.readlines()]
        self.paths = []
        self.labels = []

        for l in self.data_list:
            self.paths.append(os.path.join(self.root, l.split()[0]))
            self.labels.append(int(l.split()[1]))

        self.return_line = return_line

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert('RGB')
        lbl = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.return_line:
            return img, lbl, self.data_list[index]
        else:
            return img, lbl

    def __len__(self):
        return len(self.data_list)


class VectorDataset(Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform
        self.data_list = data_list
        self.paths = []
        self.labels = []
        for l, _ in self.data_list:
            self.paths.append(os.path.join(self.root, l.strip('\n').split()[0]))
            self.labels.append(int(l.strip('\n').split()[1]))

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.data_list[index][1], self.data_list[index][0]

    def __len__(self):
        return len(self.data_list)


# def train(args, model, device, train_loader, optimizer, iteration, epoch, prompt, lambda_=1):
#     # print('train')
#     model.train()
#     for batch_idx, (data, label) in enumerate(train_loader):
#         data, label = data.to(device), label.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.cross_entropy(output, label)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             y_prob = F.softmax(output, dim=1)
#             print(f'--> label: {label[0]}, pred: {y_prob[0].data.cpu().numpy().argmax()}, prob: {y_prob[0].data.cpu().numpy().max():.4f}')
#             print('Iteration: {}, Train on {}, Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.6f}'.format(
#                 iteration, prompt, epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))
#
#
# def train_entropy_naive(args, model, device, train_loader, optimizer, iteration, epoch, prompt, lambda_):
#     model.train()
#     for batch_idx, (data, label) in enumerate(train_loader):
#         # data, label = data.to(device), label.to(device)
#         data = data.to(device)
#         optimizer.zero_grad()
#         y = model(data)
#         y_prob = F.softmax(y, dim=1)
#         loss = -torch.sum(y_prob * torch.log(y_prob), dim=1).mean()
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print(f'--> label: {label[0]}, pred: {y_prob[0].data.cpu().numpy().argmax()}, prob: {y_prob[0].data.cpu().numpy().max():.4f}')
#             print('Iteration: {}, Train on {}, Epoch: {} [{}/{} ({:.0f}%)], Entropy: {:.6f}'.format(
#                 iteration, prompt, epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))
#
#
# def train_entropy(args, model, device, train_loader, optimizer, iteration, epoch, prompt, lambda_):
#     model.train()
#     for batch_idx, (data, prob) in enumerate(train_loader):
#         data, prob = data.to(device), prob.to(device)
#         optimizer.zero_grad()
#         y = model(data)
#         y_prob = F.softmax(y, dim=1)
#         # loss = -torch.sum(prob * torch.log(y_prob), dim=1).mean()
#
#         _, label = prob.max(dim=1)
#         loss = F.cross_entropy(y, label)
#
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             label = prob.data.cpu().numpy().argmax(axis=1)
#             print(f'--> label: {label[0]}, pred: {y_prob[0].data.cpu().numpy().argmax()}, prob: {y_prob[0].data.cpu().numpy().max():.4f}')
#             print('Iteration: {}, Train on {}, Epoch: {} [{}/{} ({:.0f}%)], Entropy: {:.6f}'.format(
#                 iteration, prompt, epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))


def train(args, model, device, train_loader, optimizer, iteration, epoch, prompt, lambda_=0.99):
    model.train()
    for batch_idx, (data, prob, _) in enumerate(train_loader):
        data, prob = data.to(device), prob.to(device)
        optimizer.zero_grad()
        y = model(data)

        _, label = prob.max(dim=1)
        loss = F.cross_entropy(y, label)
        loss.backward()

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            y_prob = F.softmax(y, dim=1)
            print(f'--> label: {label[0]}, pred: {y_prob[0].data.cpu().numpy().argmax()}, prob: {y_prob[0].data.cpu().numpy().max():.4f}')
            print('Iteration: {}, Train on {}, Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.6f}'.format(
                iteration, prompt, epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader, prompt='Test'):
    # model.eval()
    model.train()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, prob, _ in test_loader:
            data, prob = data.to(device), prob.to(device)
            y = model(data)

            _, label = prob.max(dim=1)
            test_loss += F.cross_entropy(y, label, reduction='sum').item()

            pred = y.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test on {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        prompt, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def pseudo_label(args, model, device, train_loader, iteration, selector=None, lambda_=0.99):
    print('generate softlabel')
    line_list = []
    logit_list = []

    # model.train()
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, _, line) in enumerate(train_loader):
            data = data.to(device)
            y = model(data)
            y_prob = F.softmax(y, dim=1).data.cpu()

            line_list += line
            logit_list += [y_prob.data.cpu()[ind, :] for ind in range(data.size(0))]

            if batch_idx % args.log_interval == 0:
                print('Iteration {}, Pseudo-label: [{}/{} ({:.0f}%)]'.format(
                    iteration, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader)))
        inds_sel = selector(torch.stack(logit_list))
        pseudo_list = [(line_list[ind], logit_list[ind]) for ind in inds_sel]
        print(f'Iteration {iteration}, # of pseudo-labels: {len(inds_sel)}')
    return pseudo_list


def sample_list(data_list, portion, args, seed=-1):
    if seed >= 0:
        np.random.seed(seed)
    sel_idx = np.random.choice(len(data_list), int(len(data_list) * portion), replace=False)
    new_list = list(itemgetter(*sel_idx)(data_list))
    return new_list


def datafile2onehot(datafile, args):
    with open(datafile, 'r') as f:
        data_list = f.readlines()
    new_list = []
    for l in data_list:
        l_ = l.strip('\n').split()
        L = int(l_[1])
        p = torch.zeros(args.num_classes)
        p[L] = 1.
        new_list.append((l, p))
    return new_list


def main(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

    model = CNN().to(device)

    # load pretrained model
    if args.pretrained_model_path:
        model.load_state_dict(torch.load(args.pretrained_model_path))

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    tar_epoch_cnt = 0

    src_list_full_train = datafile2onehot(args.source_list, args)
    src_list_full_test = datafile2onehot(args.source_list_test, args)
    tar_list_full_train = datafile2onehot(args.target_list, args)
    tar_list_full_test = datafile2onehot(args.target_list_test, args)
    src_loader_test = torch.utils.data.DataLoader(
        VectorDataset(args.source_root_test, src_list_full_test,
                      transforms.Compose([
                          transforms.Resize(28),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,)),
                          #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                      ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    tar_loader_train = torch.utils.data.DataLoader(
        VectorDataset(args.target_root, tar_list_full_train,
                      transforms.Compose([
                          transforms.Resize(28),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=(0.1307, 0.1307, 0.1307), std=(0.3081, 0.3081, 0.3081))
                      ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    tar_loader_test = torch.utils.data.DataLoader(
        VectorDataset(args.target_root_test, tar_list_full_test,
                      transforms.Compose([
                          transforms.Resize(28),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=(0.1307, 0.1307, 0.1307), std=(0.3081, 0.3081, 0.3081))
                      ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # eval on target
    test(args, model, device, tar_loader_test, 'Target')

    for i in range(1, args.num_iters+1):
        # generate pseudo-label
        target_portion = min(args.target_portion + (i-1) * args.target_portion_step, args.max_target_portion)
        selector = SoftmaxSelector(int(target_portion * len(tar_loader_train)),
                                   int(target_portion * len(tar_loader_train)), args.threshold)
        target_list = pseudo_label(args, model, device, tar_loader_train, i, selector=selector)  # FIXME

        # prepare mixed dataset
        source_list = sample_list(src_list_full_train, args.source_portion, args)
        source_dataset = VectorDataset(args.source_root, source_list,
                                            transforms.Compose([
                                                transforms.Resize(28),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,)),
                                                #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                            ]))
        target_dataset = VectorDataset(args.target_root, target_list,
                                            transforms.Compose([
                                                transforms.Resize(28),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.1307, 0.1307, 0.1307), std=(0.3081, 0.3081, 0.3081))
                                            ]))
        mix_loader = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset([source_dataset, target_dataset]),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        # train on source+target
        for epoch in range(1, args.num_epochs+1):
            train(args, model, device, mix_loader, optimizer, i, epoch, 'Mix')

        # test on source and target
        test(args, model, device, src_loader_test, 'Source')

        # test on source
        test(args, model, device, src_loader_test, 'Source')
        test(args, model, device, tar_loader_test, 'Target')

    if (args.save_model):
        torch.save(model.state_dict(), "mnist2mnistm_cnn.pt")


def baseline(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

    src_list_train = datafile2onehot(args.source_list, args)
    src_list_test = datafile2onehot(args.source_list_test, args)
    src_loader_train = torch.utils.data.DataLoader(
        VectorDataset(args.source_root, src_list_train,
                      transforms.Compose([
                          transforms.Resize(28),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,)),
                          #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                      ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    src_loader_test = torch.utils.data.DataLoader(
        VectorDataset(args.source_root_test, src_list_test,
                      transforms.Compose([
                          transforms.Resize(28),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,)),
                          #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                      ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    model = CNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)

    for epoch in range(1, args.num_epochs + 1):
        train(args, model, device, src_loader_train, optimizer, 0, epoch, 'Source')
        test(args, model, device, src_loader_test, 'Source')

    torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--num-epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--num-epochs-target', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--num-iters', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--source-root', type=str, default='datasets/mnist/train')
    parser.add_argument('--source-list', type=str, default='sourcefiles/mnist_train.txt')
    parser.add_argument('--source-root-test', type=str, default='datasets/mnist/test')
    parser.add_argument('--source-list-test', type=str, default='sourcefiles/mnist_test.txt')
    parser.add_argument('--target-root', type=str, default='data/mnist_m/mnist_m_train')
    parser.add_argument('--target-list', type=str, default='data/mnist_m/mnist_m_train_labels.txt')
    parser.add_argument('--target-root-test', type=str, default='data/mnist_m/mnist_m_test')
    parser.add_argument('--target-list-test', type=str, default='data/mnist_m/mnist_m_test_labels.txt')
    parser.add_argument('--T', type=int, default=1, help='number of MC samples')
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--source-portion', type=float, default=0.4)
    parser.add_argument('--source-portion-step', type=float, default=0.05)
    parser.add_argument('--max-source-portion', type=float, default=0.8)
    parser.add_argument('--target-portion', type=float, default=0.2)
    parser.add_argument('--target-portion-step', type=float, default=0.05)
    parser.add_argument('--max-target-portion', type=float, default=0.7)
    parser.add_argument('--method', type=str, default='pseudo')
    parser.add_argument('--num-workers', type=int, default=4)

    parser.add_argument('--pretrained_model_path', type=str, default='mnist_cnn.pt')
    args = parser.parse_args()

    if args.baseline:
        baseline(args)
    else:
        main(args)
