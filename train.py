import torch
import torchvision
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
from model import Net


def load_mninst(args):

    batch_size_train = 64
    batch_size_test = 1000
    random_seed = 1
    torch.manual_seed(random_seed)

    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('../data/', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
      batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('../data/', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
      batch_size=batch_size_test, shuffle=True)

    # examples = enumerate(test_loader)
    # batch_idx, (example_data, example_targets) = next(examples)
    # print(example_targets)
    # print(example_data.shape)
    #
    # fig = plt.figure()
    # for i in range(6):
    #   plt.subplot(2,3,i+1)
    #   plt.tight_layout()
    #   plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    #   plt.title("Ground Truth: {}".format(example_targets[i]))
    #   plt.xticks([])
    #   plt.yticks([])
    # plt.show()

    return train_loader, test_loader


def train(args, train_loader, test_loader, epoch):
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(args.n_epochs + 1)]
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('n_epochs', type=int, default=3)
    parser.add_argument('batch_size_train', type=int, default=64)
    parser.add_argument('batch_size_test', type=int, default=1000)
    parser.add_argument('log_interval', type=int, default=10)
    parser.add_argument('random_seed', type=int, default=1)
    learning_rate = 0.01
    momentum = 0.5
    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum)