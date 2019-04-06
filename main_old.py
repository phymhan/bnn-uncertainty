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


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


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
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        with open(data_list, 'r') as f:
            self.data_list = [l.strip('\n') for l in f.readlines()]
        self.paths = []
        self.labels = []

        for l in self.data_list:
            self.paths.append(os.path.join(self.root, l.split()[0]))
            self.labels.append(int(l.split()[1]))

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert('RGB')
        lbl = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img[1:2, ...], lbl

    def __len__(self):
        return len(self.data_list)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader, prompt='Test'):
    # model.eval()
    model.train()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        prompt, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def bayes_train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        y_prob = 0
        for t in range(args.T):
            y_pred, s_pred = model(data)

            # Gaussian noise (reparameterize)
            y_noised = reparameterize(y_pred, s_pred)

            y_prob += F.softmax(y_noised, dim=1)

        y_prob /= args.T
        # loss = F.cross_entropy(y_noised, target)
        loss = F.nll_loss(torch.log(y_prob), target)
        loss.backward()
        optimizer.step()

        # entropy
        epistemic_uncertainty = -torch.sum(y_prob * torch.log(y_prob), dim=1)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}\tEntropy: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), epistemic_uncertainty.mean().item()))


def bayes_test(args, model, device, test_loader, prompt='Test'):
    # model.eval()
    model.train()
    test_loss = 0
    correct = 0
    entropy_sum = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            y_prob = 0
            for t in range(args.T):
                y_pred, s_pred = model(data)

                # Gaussian noise (reparameterize)
                y_noised = reparameterize(y_pred, s_pred)

                y_prob += F.softmax(y_noised, dim=1)

            y_prob /= args.T
            # test_loss += F.cross_entropy(y_noised, target, reduction='sum').item()  # sum up batch loss
            test_loss += F.nll_loss(torch.log(y_prob), target, reduction='sum').item()
            pred = y_noised.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            # entropy
            epistemic_uncertainty = -torch.sum(y_prob * torch.log(y_prob), dim=1)
            entropy_sum += epistemic_uncertainty.mean().item()

    test_loss /= len(test_loader.dataset)

    print('{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Entropy: {:.4f}\n'.format(
        prompt, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), entropy_sum/len(test_loader)))


def bayes_train_da(args, model, device, train_loader, target_loader, optimizer, epoch):
    model.train()
    target_iter = iter(target_loader)
    print('-------------')
    print(len(target_iter))
    print(len(target_loader))

    alpha = 2. / (1. + np.exp(-10 * epoch/args.epochs)) - 1

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        y_prob = 0
        for t in range(args.T):
            y_pred, s_pred = model(data)

            # Gaussian noise (reparameterize)
            y_noised = reparameterize(y_pred, s_pred)

            y_prob += F.softmax(y_noised, dim=1)

        y_prob /= args.T
        # loss = F.cross_entropy(y_noised, target)
        loss_nll = F.nll_loss(torch.log(y_prob), target)

        # entropy on train data
        entropy_source = -torch.sum(y_prob * torch.log(y_prob), dim=1)

        # entropy on target data
        if batch_idx >= len(target_loader):
            target_iter = iter(target_loader)
        data_target, _ = target_iter.next()

        data_target = data_target.to(device)
        y_prob = 0
        for t in range(args.T):
            y_pred, s_pred = model(data_target)
            y_noised = reparameterize(y_pred, s_pred)
            y_prob += F.softmax(y_noised, dim=1)
        entropy_target = -torch.sum(y_prob * torch.log(y_prob), dim=1)
        loss_domain = (entropy_target.mean()-entropy_source.mean()).pow(2)

        loss = loss_nll + 0.01*alpha*loss_domain

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}\tTarget Entropy: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), entropy_target.mean().item()))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--num-epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--target-root', type=str, default='data/mnist_m/mnist_m_train')
    parser.add_argument('--target-list', type=str, default='data/mnist_m/mnist_m_train_labels.txt')
    parser.add_argument('--T', type=int, default=1, help='number of MC samples')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/mnist', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    target_loader = torch.utils.data.DataLoader(
        ImageFolderDataset(args.target_root, args.target_list,
                           transforms.Compose([
                               transforms.Resize(28),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                           ])),
        batch_size=args.batch_size, shuffle=True, num_workers=8
    )

    model = CNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)

    for epoch in range(1, args.num_epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader, 'Test')
        test(args, model, device, target_loader, 'Target')

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
