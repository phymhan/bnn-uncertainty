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
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
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
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
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
            return img[1:2, ...], lbl, self.data_list[index]
        else:
            return img[1:2, ...], lbl

    def __len__(self):
        return len(self.data_list)


class EmbeddingDataset(Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform
        self.data_list = data_list
        self.paths = []
        for l, _ in self.data_list:
            self.paths.append(os.path.join(self.root, l.split()[0]))

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img[1:2, ...], self.data_list[index][1]

    def __len__(self):
        return len(self.data_list)


def train(args, model, device, train_loader, optimizer, iteration, epoch, prompt, lambda_=1):
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, label)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            y_prob = F.softmax(output, dim=1)
            print(f'--> label: {label[0]}, pred: {y_prob[0].data.cpu().numpy().argmax()}, prob: {y_prob[0].data.cpu().numpy().max():.4f}')
            print('Iteration: {}, Train on {}, Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.6f}'.format(
                iteration, prompt, epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def train_entropy_naive(args, model, device, train_loader, optimizer, iteration, epoch, prompt, lambda_):
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        # data, label = data.to(device), label.to(device)
        data = data.to(device)
        optimizer.zero_grad()
        y = model(data)
        y_prob = F.softmax(y, dim=1)
        loss = -torch.sum(y_prob * torch.log(y_prob), dim=1).mean()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(f'--> label: {label[0]}, pred: {y_prob[0].data.cpu().numpy().argmax()}, prob: {y_prob[0].data.cpu().numpy().max():.4f}')
            print('Iteration: {}, Train on {}, Epoch: {} [{}/{} ({:.0f}%)], Entropy: {:.6f}'.format(
                iteration, prompt, epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# def train_entropy(args, model, device, train_loader, optimizer, iteration, epoch, prompt, lambda_):
#     model.train()
#     for batch_idx, (data, label) in enumerate(train_loader):
#         # data, label = data.to(device), label.to(device)
#         data = data.to(device)
#         optimizer.zero_grad()
#         y = model(data)
#
#         y_ = y.detach()
#         y_anneal = F.softmax(y_/(1-lambda_), dim=1)
#
#         y_prob = F.softmax(y, dim=1)
#         loss = -torch.sum(y_anneal * torch.log(y_prob), dim=1).mean()
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print(f'--> label: {label[0]}, pred: {y_prob[0].data.cpu().numpy().argmax()}, prob: {y_prob[0].data.cpu().numpy().max():.4f}')
#             print('Iteration: {}, Train on {}, Epoch: {} [{}/{} ({:.0f}%)], Entropy: {:.6f}'.format(
#                 iteration, prompt, epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))


def train_entropy(args, model, device, train_loader, optimizer, iteration, epoch, prompt, lambda_):
    model.train()
    for batch_idx, (data, prob) in enumerate(train_loader):
        data, prob = data.to(device), prob.to(device)
        optimizer.zero_grad()
        y = model(data)
        y_prob = F.softmax(y, dim=1)
        # loss = -torch.sum(prob * torch.log(y_prob), dim=1).mean()

        _, label = prob.max(dim=1)
        loss = F.cross_entropy(y, label)

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            label = prob.data.cpu().numpy().argmax(axis=1)
            print(f'--> label: {label[0]}, pred: {y_prob[0].data.cpu().numpy().argmax()}, prob: {y_prob[0].data.cpu().numpy().max():.4f}')
            print('Iteration: {}, Train on {}, Epoch: {} [{}/{} ({:.0f}%)], Entropy: {:.6f}'.format(
                iteration, prompt, epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def train_pseudo_online(args, model, device, train_loader, optimizer, iteration, epoch, prompt, lambda_):
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        # data, label = data.to(device), label.to(device)
        data = data.to(device)
        optimizer.zero_grad()
        y = model(data)

        _, label_online = torch.max(y, dim=1)
        label_online = label_online.to(device)
        loss = F.cross_entropy(y, label_online.detach())
        loss.backward()

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            y_prob = F.softmax(y, dim=1)
            print(f'>> online label acc {accuracy_score(label.detach().data.cpu().numpy(), label_online.detach().data.cpu().numpy()) * 100:.2f}')
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
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += F.cross_entropy(output, label, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test on {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        prompt, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def gen_label(args, model, device, train_loader, iteration, selector=None):
    pseudo_list = []

    # model.train()
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, label, line) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            y = model(data)
            inds = selector(y)
            pseudo_list += [line[ind] for ind in inds]

            # y_prob = F.softmax(y, dim=1)
            # print(f'--> label: {label[0]}, pred: {y_prob[0].data.cpu().numpy().argmax()}, prob: {y_prob[0].data.cpu().numpy().max():.4f}')

            if batch_idx % args.log_interval == 0:
                print('Iteration {}, Pseudo-label: [{}/{} ({:.0f}%)]'.format(
                    iteration, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader)))
        print(f'Iteration {iteration}, # of pseudo-labels: {len(pseudo_list)}')
    return pseudo_list


def gen_softlabel(args, model, device, train_loader, iteration, selector=None, lambda_=0.):
    pseudo_list = []

    # model.train()
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, label, line) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            y = model(data)
            y_anneal = F.softmax(y.detach() / (1 - lambda_), dim=1).data.cpu().numpy()
            inds = selector(y)
            pseudo_list += [(line[ind], y_anneal[ind]) for ind in inds]

            # y_prob = F.softmax(y, dim=1)
            # print(f'--> label: {label[0]}, pred: {y_prob[0].data.cpu().numpy().argmax()}, prob: {y_prob[0].data.cpu().numpy().max():.4f}')

            if batch_idx % args.log_interval == 0:
                print('Iteration {}, Pseudo-label: [{}/{} ({:.0f}%)]'.format(
                    iteration, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader)))
        print(f'Iteration {iteration}, # of pseudo-labels: {len(pseudo_list)}')
    return pseudo_list


def main_old():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--num-epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--num-epochs-target', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--num-iters', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--target-root', type=str, default='data/mnist_m/mnist_m_train')
    parser.add_argument('--target-list', type=str, default='data/mnist_m/mnist_m_train_labels.txt')
    parser.add_argument('--target-root-test', type=str, default='data/mnist_m/mnist_m_test')
    parser.add_argument('--target-list-test', type=str, default='data/mnist_m/mnist_m_test_labels.txt')
    parser.add_argument('--T', type=int, default=1, help='number of MC samples')
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--min_kept_ratio', type=float, default=0.1)
    parser.add_argument('--method', type=str, default='pseudo')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    src_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    src_loader_test = torch.utils.data.DataLoader(
        datasets.MNIST('data/mnist', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    tar_loader = torch.utils.data.DataLoader(
        ImageFolderDataset(args.target_root, args.target_list,
                           transforms.Compose([
                               transforms.Resize(28),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                           ]),
                           return_line=True),
        batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=True
    )  # FIXME
    tar_loader_test = torch.utils.data.DataLoader(
        ImageFolderDataset(args.target_root_test, args.target_list_test,
                           transforms.Compose([
                               transforms.Resize(28),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                           ])),
        batch_size=args.batch_size, shuffle=True, num_workers=8
    )

    if args.method == 'pseudo':
        selector = SoftmaxSelector(int(args.min_kept_ratio*args.batch_size), 0, args.threshold)
        # train_target = train_pseudo_online
        train_target = train
    else:
        # selector = EntropySelector(int(args.min_kept_ratio*args.batch_size), 0, args.threshold)
        selector = SoftmaxSelector(int(args.min_kept_ratio * args.batch_size), 0, args.threshold)
        train_target = train_entropy

    model = CNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    tar_epoch_cnt = 0

    for i in range(1, args.num_iters+1):
        # train on source
        for epoch in range(1, args.num_epochs+1):
            train(args, model, device, src_loader, optimizer, i, epoch, 'Source')

        # test on source
        test(args, model, device, src_loader_test, 'Source')

        # pseudo label
        pseudo_list = gen_label(args, model, device, tar_loader, i, selector=selector)
        tar_loader_train = torch.utils.data.DataLoader(
            ImageFolderDataset(args.target_root, pseudo_list,
                               transforms.Compose([
                                   transforms.Resize(28),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                               ])),
            batch_size=args.batch_size, shuffle=True, num_workers=8
        )  # FIXME

        # train on target
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = param_group['lr'] * 2
        for epoch in range(1, args.num_epochs_target+1):
            tar_epoch_cnt += 1
            lambda_ = 2. / (1. + np.exp(-10 * tar_epoch_cnt/(args.num_epochs_target*args.num_iters))) - 1
            # print(f'==> lambda = {lambda_:.4f}')
            train_target(args, model, device, tar_loader_train, optimizer, i, epoch, 'Target', 0.99)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = args.lr

        # test on target
        test(args, model, device, tar_loader_test, 'Target')

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--num-epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--num-epochs-target', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--num-iters', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--target-root', type=str, default='data/mnist_m/mnist_m_train')
    parser.add_argument('--target-list', type=str, default='data/mnist_m/mnist_m_train_labels.txt')
    parser.add_argument('--target-root-test', type=str, default='data/mnist_m/mnist_m_test')
    parser.add_argument('--target-list-test', type=str, default='data/mnist_m/mnist_m_test_labels.txt')
    parser.add_argument('--T', type=int, default=1, help='number of MC samples')
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--min_kept_ratio', type=float, default=0.1)
    parser.add_argument('--method', type=str, default='pseudo')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    src_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    src_loader_test = torch.utils.data.DataLoader(
        datasets.MNIST('data/mnist', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    tar_loader = torch.utils.data.DataLoader(
        ImageFolderDataset(args.target_root, args.target_list,
                           transforms.Compose([
                               transforms.Resize(28),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                           ]),
                           return_line=True),
        batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=True
    )  # FIXME
    tar_loader_test = torch.utils.data.DataLoader(
        ImageFolderDataset(args.target_root_test, args.target_list_test,
                           transforms.Compose([
                               transforms.Resize(28),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                           ])),
        batch_size=args.batch_size, shuffle=True, num_workers=8
    )

    if args.method == 'pseudo':
        selector = SoftmaxSelector(int(args.min_kept_ratio*args.batch_size), 0, args.threshold)
        # train_target = train_pseudo_online
        train_target = train
    else:
        # selector = EntropySelector(int(args.min_kept_ratio*args.batch_size), 0, args.threshold)
        selector = SoftmaxSelector(int(args.min_kept_ratio * args.batch_size), 0, args.threshold)
        train_target = train_entropy

    model = CNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    tar_epoch_cnt = 0

    for i in range(1, args.num_iters+1):
        # train on source
        for epoch in range(1, args.num_epochs+1):
            train(args, model, device, src_loader, optimizer, i, epoch, 'Source')

        # test on source
        test(args, model, device, src_loader_test, 'Source')

        # pseudo label
        pseudo_list = gen_softlabel(args, model, device, tar_loader, i, selector=selector, lambda_=0.99999)  # FIXME: fix lambda=0.99 here, change to store logits
        tar_loader_train = torch.utils.data.DataLoader(
            EmbeddingDataset(args.target_root, pseudo_list,
                               transforms.Compose([
                                   transforms.Resize(28),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                               ])),
            batch_size=args.batch_size, shuffle=True, num_workers=8
        )  # FIXME

        # train on target
        for epoch in range(1, args.num_epochs_target+1):
            tar_epoch_cnt += 1
            lambda_ = 2. / (1. + np.exp(-10 * tar_epoch_cnt/(args.num_epochs_target*args.num_iters))) - 1
            # print(f'==> lambda = {lambda_:.4f}')
            train_target(args, model, device, tar_loader_train, optimizer, i, epoch, 'Target', 0.99)

        # test on target
        test(args, model, device, tar_loader_test, 'Target')

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main_old()
