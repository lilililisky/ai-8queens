"""
极简卷积神经网络 (CNN) - 基于公众号文章实现
用于 MNIST 手写数字识别
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

# 设置设备（优先使用 GPU，如果没有则使用 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 超参数
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# 数据预处理：转换为张量并归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 下载 MNIST 数据集（自动下载，约 50MB）
print("正在下载/加载 MNIST 数据集...")
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")


# 定义极简 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 卷积层1：1→32通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 卷积层2：32→64通道
        self.pool = nn.MaxPool2d(2, 2)  # 池化层：2x2 最大池化
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 全连接层1
        self.fc2 = nn.Linear(128, 10)  # 输出层（10个数字类别）
        self.relu = nn.ReLU()

    def forward(self, x):
        # 输入: (batch, 1, 28, 28)
        x = self.pool(self.relu(self.conv1(x)))  # -> (batch, 32, 14, 14)
        x = self.pool(self.relu(self.conv2(x)))  # -> (batch, 64, 7, 7)
        x = x.view(-1, 64 * 7 * 7)  # 展平
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleCNN().to(device)
print("\n模型结构:")
print(model)

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params:,}")

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# 训练函数
def train():
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (i + 1) % 200 == 0:
            print(f"  批次 [{i+1}/{len(train_loader)}], "
                  f"Loss: {running_loss/(i+1):.4f}, "
                  f"Acc: {100.*correct/total:.2f}%")

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# 测试函数
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    return accuracy


# 开始训练
print("\n" + "="*50)
print("开始训练 SimpleCNN")
print("="*50)

for epoch in range(1, EPOCHS+1):
    start_time = time.time()
    train_loss, train_acc = train()
    test_acc = test()
    elapsed = time.time() - start_time

    print(f"\nEpoch [{epoch}/{EPOCHS}]")
    print(f"  训练 Loss: {train_loss:.4f}, 训练 Acc: {train_acc:.2f}%")
    print(f"  测试 Acc: {test_acc:.2f}%")
    print(f"  耗时: {elapsed:.2f}秒")
    print("-"*30)

print("\n✅ 极简 CNN 训练完成！")