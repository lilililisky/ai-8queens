"""
LeNet-5 卷积神经网络 - 用于 MNIST 手写数字识别
经典 LeNet-5 结构的变体（适配 MNIST 尺寸）
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 超参数
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载数据
print("正在加载 MNIST 数据集...")
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 定义 LeNet-5 模型（适配 28x28 输入）
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 卷积层1: 输入1通道 → 6通道, 5x5卷积核
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        # 池化层1: 2x2 平均池化
        self.pool1 = nn.AvgPool2d(2, 2)
        # 卷积层2: 6通道 → 16通道, 5x5卷积核
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # 池化层2: 2x2 平均池化
        self.pool2 = nn.AvgPool2d(2, 2)
        # 全连接层1: 16*5*5 → 120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 全连接层2: 120 → 84
        self.fc2 = nn.Linear(120, 84)
        # 输出层: 84 → 10
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 输入: (batch, 1, 28, 28)
        x = self.relu(self.conv1(x))   # -> (batch, 6, 28, 28)
        x = self.pool1(x)               # -> (batch, 6, 14, 14)
        x = self.relu(self.conv2(x))   # -> (batch, 16, 10, 10)
        x = self.pool2(x)               # -> (batch, 16, 5, 5)
        x = x.view(-1, 16 * 5 * 5)     # 展平: (batch, 400)
        x = self.relu(self.fc1(x))     # -> (batch, 120)
        x = self.relu(self.fc2(x))     # -> (batch, 84)
        x = self.fc3(x)                # -> (batch, 10)
        return x


model = LeNet5().to(device)
print("\n模型结构:")
print(model)

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params:,}")

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


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

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


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
print("开始训练 LeNet-5")
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

print("\n✅ LeNet-5 训练完成！")