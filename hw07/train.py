import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

# 参数
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001
IMG_SIZE = 224

# 数据路径（改成你的路径）
DATA_DIR = "chest_xray/chest_xray"

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载数据
full_train = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
train_size = int(0.8 * len(full_train))
val_size = len(full_train) - train_size
train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

class_names = full_train.classes
print(f"类别: {class_names}")
print(f"训练: {len(train_dataset)}, 验证: {len(val_dataset)}, 测试: {len(test_dataset)}")

# 模型
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

# 训练
print("\n开始训练...")
for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss, train_correct = 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += (out.argmax(1) == y).sum().item()
    
    # 验证
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            val_correct += (out.argmax(1) == y).sum().item()
    
    train_acc = 100 * train_correct / len(train_dataset)
    val_acc = 100 * val_correct / len(val_dataset)
    print(f"Epoch {epoch}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

# 测试
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        out = model(x)
        y_true.extend(y.numpy())
        y_pred.extend(out.argmax(1).cpu().numpy())

acc = accuracy_score(y_true, y_pred)
print(f"\n测试准确率: {acc:.4f} ({acc*100:.2f}%)")
print(classification_report(y_true, y_pred, target_names=class_names))

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()