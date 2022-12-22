import torch
import torchvision
import torchvision.transforms as transforms
from repvgg import *
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import time


def get_dataloader(batch_size):
    # 数据操作
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(150),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     # 规整张量
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

        "val": transforms.Compose([transforms.Resize([150, 150]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    # 加载数据
    train_dataset = torchvision.datasets.ImageFolder('dataset/train',
                                                     transform=data_transform["train"])
    test_dataset = torchvision.datasets.ImageFolder('dataset/val',
                                                    transform=data_transform["val"])
    print(f'训练数据集长度为：{len(train_dataset)}')
    print(f'测试数据集长度为：{len(test_dataset)}')

    # 创建DataLoader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)

    return train_dataloader, test_dataloader


# 添加TensorBoard
writer = SummaryWriter('./log_repvggA1_IntelImageClassification')

# 设置device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载网络
deploy = False
use_checkpoint = False
net = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=6,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint)
net = net.to(device=device)

# 设置损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device=device)

# 设置优化器
learing_rate = 1e-3
optimizer = torch.optim.Adam(params=net.parameters(),lr=learing_rate)

# 设置学习率调整策略
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.1)

# 设置训练轮次
num_epoch = 200

# 获取Dataloader
batch_size = 128
train_dataloader, test_dataloader = get_dataloader(batch_size=batch_size)

# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0


best_acc = 0.0

start_time = time.time()
for epoch in range(num_epoch):
    print("——————第 {} 轮训练开始——————".format(epoch + 1))

    # 训练
    net.train()
    train_acc = 0
    for batch in tqdm(train_dataloader, desc='训练'):
        imgs, targets = batch
        imgs = imgs.to(device=device)
        targets = targets.to(device=device)
        output = net(imgs)

        # 计算损失
        Loss = loss_fn(output, targets)

        #  优化更新
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()

        # 获取训练中的预测值
        # 注意获取的是最大值的下标值
        _, pred = output.max(1)

        num_correct = (pred == targets).sum().item()
        # 一个批量的准确率
        acc = num_correct / batch_size
        # 一个epoch的总准确率
        train_acc += acc

    # 输出一次epoch训练后的结果
    print("epoch: {}, Loss: {}, Acc: {}".format(epoch,
                                                Loss.item(),
                                                train_acc / len(train_dataloader)))
    writer.add_scalar('train_Loss', Loss.item(), epoch + 1)
    writer.add_scalar('train_Acc', train_acc / len(train_dataloader), epoch + 1)

    # 测试
    net.eval()
    eval_loss = 0
    eval_acc = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='测试'):
            imgs, targets = batch
            imgs = imgs.to(device=device)
            targets = targets.to(device=device)
            output = net(imgs)
            Loss = loss_fn(output, targets)
            _, pred = output.max(1)
            num_correct = (pred == targets).sum().item()
            eval_loss += Loss
            acc = num_correct / batch_size
            eval_acc += acc

        eval_losses = eval_loss / (len(test_dataloader))
        eval_acc = eval_acc / (len(test_dataloader))
        torch.save(net.state_dict(), f'./save_pth/epoch{epoch + 1}.pth')
        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(net.state_dict(), './save_pth/best/best.pth')

        print(f"验证集上的Loss: {eval_losses}")
        print(f"验证集上的正确率: {eval_acc}")
        writer.add_scalar('val_Loss', eval_losses, epoch + 1)
        writer.add_scalar('val_Acc', eval_acc, epoch + 1)
        scheduler.step()
writer.close()
print(f'最佳精度：{best_acc}')
total_time = time.time() - start_time
print('Training complete in {:.0f}m {:.0f}s'.format(
    total_time // 60, total_time % 60))
