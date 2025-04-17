import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
import logging
import torch.nn.functional as F
import random

# 减小批量大小
batch_size = 64
seed = 42
num_epochs = 500
lr = 9e-4
min_lr = 1e-6  # 最小学习率
loss_history = []  # 损失历史记录

# 禁用 cuDNN 的自动寻找算法功能
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed)


class ImageFileDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: 包含图像的文件夹路径
        transform: 应用于图像的可选转换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('L')  # 确保图像为灰度格式

        if self.transform:
            image = self.transform(image)

        # 从文件名中提取标签
        label_name = os.path.splitext(img_name)[0]
        label_parts = label_name.split('_')
        if len(label_parts) != 8:
            label_parts = label_parts[:-1]
        label = float(label_parts[para])
        label = torch.tensor(label, dtype=torch.float32)
        return image, label


# 数据加载和预处理
def load_train_data():
    """
    加载 BH 数据集并进行预处理
    :return: 训练数据加载器
    """
    # 定义图像预处理流程
    # 设置数据转换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像尺寸
        transforms.ToTensor(),  # 转换为 PyTorch 张量
        transforms.Normalize((0.5,), (0.5,))  # 归一化
    ])

    # 创建自定义数据集实例
    train_dataset = ImageFileDataset(root_dir=f'./{DataPath}/train', transform=transform)
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 调试信息，打印特征图的通道数
    for images, _ in train_loader:
        print(f"Input channels: {images.shape[1]}")
        break

    return train_loader


# 数据加载和预处理
def load_val_data():
    """
    加载 BH 数据集并进行预处理
    :return: 验证数据加载器
    """
    # 定义图像预处理流程
    # 设置数据转换
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像尺寸
        transforms.ToTensor(),  # 转换为 PyTorch 张量
        transforms.Normalize((0.5,), (0.5,))  # 归一化
    ])

    # 创建自定义数据集实例
    val_dataset = ImageFileDataset(root_dir=f'./{DataPath}/val', transform=transform)
    # 创建数据加载器
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return val_loader


# 多尺度特征模块
class MultiScaleFeatures(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFeatures, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels // 3, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels // 3, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels - 2 * (out_channels // 3), kernel_size=5, padding=2)
        self.fusion = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.conv5x5(x)
        out = torch.cat([x1, x2, x3], dim=1)
        out = self.fusion(out)
        return out


# 构建模型
class RegressionResNet(models.ResNet):
    def __init__(self):
        super(RegressionResNet, self).__init__(models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=1)
        # 修改第一个卷积层的输入通道数为 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.fc.in_features, 1)
        # 加入多尺度特征模块
        self.multi_scale = MultiScaleFeatures(64, 64)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 多尺度特征提取
        x = self.multi_scale(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def r2_score(y_true, y_pred):
    # 假设 y_true 是真实值的数组，y_pred 是预测值的数组
    # 计算 SST
    y_mean = np.mean(y_true)
    SST = np.sum((y_true - y_mean) ** 2)
    # 计算 SSR
    SSR = np.sum((y_true - y_pred) ** 2)
    # 计算 R²
    R_squared = 1 - SSR / SST
    return R_squared


# 测试模型并计算 R² 分数
def val(valloader):
    model.eval()
    with torch.no_grad():
        for batch, (images, labels) in enumerate(valloader):
            images = images.to(device)
            labels = labels.to(device).float()
            outputs = model(images)
            if para == 6:
                # 获取模型输出的最后一个特征，并进行阈值判断
                last_feature_outputs = outputs
                thresholded_outputs = (last_feature_outputs > 0).float() * 1.033782 + (
                            last_feature_outputs <= 0).float() * -0.966816
                # 替换 outputs 中的最后一个特征
                outputs = thresholded_outputs
            labels = labels.unsqueeze(1).detach().cpu().numpy().flatten()
            outputs = outputs.detach().cpu().numpy().flatten()
        global r2_best
        r2 = r2_score(labels, outputs)
        if r2 > r2_best:
            model_path = os.path.join("result_for_Single-parameter\\result\saved_models",
                                      f"para{para}_model_epoch_best.pth")
            torch.save(model.state_dict(), model_path)
            r2_best = r2
        r2_scores.append(r2)
        print(f'R² Score: {r2:.4f}')


os.makedirs('result_for_Single-parameter\\result', exist_ok=True)
os.makedirs('result_for_Single-parameter\\result\saved_models', exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('result_for_Single-parameter\\result\model_training.log'),  # 将日志写入文件
        logging.StreamHandler()  # 同时在控制台输出日志
    ]
)


# 训练模型
def train(trainloader):
    epoch_losses = []
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 使用余弦退火调度学习率
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch, (images, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device).float()  # 确保标签是浮点数
            labels = labels.unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            loss.backward()
            optimizer.step()
        loss_history.append(running_loss / len(trainloader.dataset))

        epoch_loss = running_loss / len(trainloader.dataset)
        epoch_losses.append(2 if epoch_loss > 2 else epoch_loss)
        # 使用 logging 记录信息
        logging.info(
            f"Epoch : {epoch + 1}/{num_epochs} - loss : {epoch_loss:.4f} - lr : {scheduler.get_last_lr()[0]}"
        )
        scheduler.step()  # 更新学习率
        if (epoch + 1) % 10 == 0:
            valloader = load_val_data()
            val(valloader)
        # 释放显存
        torch.cuda.empty_cache()
    # 绘制损失变化曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制 R² 分数曲线
    plt.subplot(1, 2, 2)
    plt.plot(r2_scores, label=f'R² Score')
    plt.xlabel('Epochs')
    plt.ylabel('R² Score')
    plt.legend()
    plt.savefig(f'result_for_Single-parameter\\result/para{para}_Training_Loss_R2_plots.png')

    with open(f'result_for_Single-parameter\\result/para{para}_epoch_losses.txt', 'w') as file:
        for epoch, loss in enumerate(epoch_losses):
            file.write(f'Epoch {epoch + 1}: Loss = {loss}\n')

    # 将 r2_scores 写入文本文件
    with open(f'result_for_Single-parameter\\result/para{para}_r2_scores.txt', 'w') as file:
        for epoch, score in enumerate(r2_scores):
            file.write(f'Epoch {epoch + 1}: R2 Score = {score}\n')
        file.write(f'MAX: R2 Score = {max(r2_scores)}\n')


paras = [0, 2, 3, 4, 5, 7]
DataPath = "GRRT_dataset"
r2_best = -np.inf
r2_scores = []

model = RegressionResNet()
# 移动模型到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# 损失函数
criterion = nn.MSELoss()

for para in paras:
    print(f"第{para + 1}个参数训练开始")
    trainloader = load_train_data()
    train(trainloader)
    r2_scores = []
    r2_best = -np.inf
    model = RegressionResNet()
    model = model.to(device)
    # 损失函数和优化器
    criterion = nn.MSELoss()
    loss_history = []
