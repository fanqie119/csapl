import os
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import argparse
from torch.utils.tensorboard import SummaryWriter

from utils.utils_data import get_params_groups, create_lr_scheduler
from torch.utils.data import DataLoader
from model import Net


def load_mninst(batch_size_train, batch_size_test, random_seed):
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


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):

    train_losses = []
    loss_function = torch.nn.CrossEntropyLoss()
    accu_num = torch.zeros(1).to(device)
    sample_num = 0
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        sample_num += data.shape[0]
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        output.data.max(1, keepdim=True)
        pred = output.data.max(1, keepdim=True)[1]
        loss = loss_function(pred, target.to(device))
        accu_num += torch.eq(pred, target.to(device)).sum()
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(data_loader.dataset),
                    100. * batch_idx / len(data_loader), loss.item()))
        train_losses.append(loss.item())

    return train_losses, accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):

    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}, top_acc:{:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
        )

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if os.path.exists("./model") is False:
        os.makedirs("./model")

    train_loader, test_loader = load_mninst(args.batch_size_train, args.batch_size_test, args.random_seed)
    model = Net().to(device)

    # pg = get_params_groups(model, weight_decay=args.wd)
    # optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    # lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
    #                                    warmup=True, warmup_epochs=1)

    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=args.learning_rate,
                          momentum=args.momentum)

    best_acc = 0.
    tb_writer = SummaryWriter()
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=args.learning_rate)

        # # validate
        # val_loss, val_acc = evaluate(model=model,
        #                              data_loader=val_loader,
        #                              device=device,
        #                              epoch=epoch)

        # tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        # tb_writer.add_scalar(tags[0], train_loss, epoch)
        # tb_writer.add_scalar(tags[1], train_acc, epoch)
        # tb_writer.add_scalar(tags[2], val_loss, epoch)
        # tb_writer.add_scalar(tags[3], val_acc, epoch)
        # tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if best_acc < train_acc:
            # torch.save(model.state_dict(), "./weights/best_model.pth")
            torch.save(model.state_dict(), './model/model.pth')
            torch.save(optimizer.state_dict(), './model/optimizer.pth')
            # best_acc = val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size_train', type=int, default=64)
    parser.add_argument('--batch_size_test', type=int, default=1000)
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--wd', type=float, default=5e-2)

    opt = parser.parse_args()
    main(opt)

