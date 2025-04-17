import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
import logging
import torch.nn.functional as F
import random

batch_size = 256
seed = 42
num_epochs = 500
lr = 1e-3
gamma = 0.95

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
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        label_name = os.path.splitext(img_name)[0]
        label_parts = label_name.split('_')
        label = float(label_parts[para])
        label = torch.tensor(label, dtype=torch.float32)
        return image, label

def load_train_data():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = ImageFileDataset(root_dir=f'./{DataPath}/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

def load_test_data():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = ImageFileDataset(root_dir=f'./{DataPath}/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return test_loader

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
    y_mean = np.mean(y_true)
    SST = np.sum((y_true - y_mean) ** 2)
    SSR = np.sum((y_true - y_pred) ** 2)
    R_squared = 1 - SSR / SST
    return R_squared

def test(testloader, para):
    model.eval()
    r2_scores = []
    stds_list = []
    with torch.no_grad():
        for batch, (images, labels) in enumerate(testloader):
            images = images.to(device)
            labels = labels.to(device).float()
            outputs = model(images)
            if para == 6:
                last_feature_outputs = outputs
                thresholded_outputs = (last_feature_outputs > 0).float() * 1.033782 + (last_feature_outputs <= 0).float() * -0.966816
                outputs = thresholded_outputs
            labels = labels.unsqueeze(1).detach().cpu().numpy().flatten()
            outputs = outputs.detach().cpu().numpy().flatten()
            r2 = r2_score(labels, outputs)
            r2_scores.append(r2)
            print(f'R² Score: {r2:.4f}')
            predictions = []
            label_list = []
            for i in range(len(outputs)):
                predictions.append(outputs[i].item())
                label_list.append(labels[i])

            sorted_indices = np.argsort(label_list)
            predictions = [predictions[i] for i in sorted_indices]
            label_list = [label_list[i] for i in sorted_indices]
            predictions = np.array(predictions)
            label_list = np.array(label_list)

            plt.figure(figsize=(10, 5))
            # 计算预测值和真实值之间的误差
            # plt.plot(np.arange(len(predictions)), predictions, '#015876', linewidth=2)
            # plt.plot(np.arange(len(label_list)), label_list, '#FDB137', linewidth=2)
            # plt.fill_between(np.arange(len(predictions)), lower_bounds, upper_bounds, facecolor='#C1D1D1', alpha=0.5)
            # if para == 6:
            plt.scatter(np.arange(len(predictions)), predictions, color='#015876', s=10, label='Predictions')  # s 控制点的大小
            # else:
            #     plt.errorbar(np.arange(len(predictions)), predictions, yerr=[predictions - lower_bounds, upper_bounds - predictions],
            #  fmt='.', color='#015876', ecolor='#C1D1D1', elinewidth=1, capsize=0, alpha=0.5, label='Predictions')
            plt.scatter(np.arange(len(label_list)), label_list, color='#FDB137', s=10, label='Labels')  # s 控制点的大小
            for i in range(len(predictions)):
                plt.plot([i, i], [predictions[i], label_list[i]], color='#C1D1D1', linestyle='-', linewidth=1)
            # 绘制预测值的置信区间
            plt.title(f"para{para}")
            plt.xlabel('sample')
            plt.ylabel('number')
            plt.ylim(-2.5, 2.5)
            plt.legend()
            plt.savefig(f'result_for_Single-parameter/test_result/para{para}_test_Predictions.png', dpi=300)
            plt.clf()

            # plt.figure(figsize=(6, 6))
            # correct_count = 0
            # total_count = len(predictions)
            # # 绿色点表示预测范围内，红色点表示预测范围外
            # plt.scatter(-1.7, -1.7, color='#00ECC2', s=10, label='Correct Prediction')  # 绿色点
            # correct_count += 1
            # plt.scatter(label_list[0], predictions[0], color='#FF4359', s=10, label='Incorrect Prediction')  # 红色点

            # # 然后绘制剩余的点，不设置 label
            # for label, prediction, lower_bound, upper_bound in zip(label_list[1:], predictions[1:], lower_bounds[1:], upper_bounds[1:]):
            #     if label >= lower_bound and label <= upper_bound:
            #         plt.scatter(label, prediction, color='#00ECC2', s=10)  # 绿色点
            #         correct_count += 1
            #     else:
            #         plt.scatter(label, prediction, color='#FF4359', s=10)  # 红色
            # plt.grid(color='#7d7f7c', linestyle='-.')
            # plt.plot([-2, 2], [-2, 2], 'k--', lw=2)  # 对角线
            # # 添加图例，确保包含所有标签
            # plt.legend(['Correct Prediction', 'Incorrect Prediction'])

            # plt.title(f"para{para}")
            # plt.xlabel('True Labels')
            # plt.ylabel('Predictions')
            # plt.ylim(-2.5, 2.5)
            # plt.savefig(f'result_for_Single-parameter/test_result/para{para}_scatter_test_Predictions.png', dpi=300)
            # plt.clf()
            # accuracy = correct_count / total_count
            # print(f'para{para} 正确率: {accuracy:.4f}')
            # mean_std = np.mean(stds_list)  # 计算标准差的均值
            # print(f'Mean Standard Deviation for para{para}: {mean_std:.4f}')
    return r2_scores

os.makedirs('result_for_Single-parameter/test_result', exist_ok=True)

paras = [0,2,3,4,5,7]
DataPath = "GRRT_dataset"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

r2_scores_all = []

for para in paras:
    print(f"第{para+1}个参数训测试开始")
    testloader = load_test_data()
    model = RegressionResNet()
    model = model.to(device)
    model.load_state_dict(torch.load(f'result_for_Single-parameter/result/saved_models/para{para}_model_epoch_best.pth'))
    r2_scores = test(testloader, para)
    r2_scores_all.append(r2_scores[0])  # 假设每个参数只计算一次R²分数

# 绘制R²分数的柱状图
colors = [
    'skyblue',
    '#FFA500',  # 浅橙色
    'lightgreen',
    '#FF8080',  # 浅红色
    '#E6E6FA',  # 浅紫色
    '#F4A460',  # 浅褐色
    '#FFB6C1',  # 浅粉色
    'lightgray'
]

plt.figure(figsize=(10, 6))

# 绘制柱状图，并为每个柱状图指定颜色
bars = plt.bar(range(len(paras)), r2_scores_all, color=colors)

# 在每个柱状图上标出分数
for bar, score in zip(bars, r2_scores_all):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, format(score, '.4f'), ha='center', va='bottom')

plt.title('R² Scores for Different Parameters')
plt.xlabel('Parameter Index')
plt.ylabel('R² Score')
plt.xticks(range(len(paras)), paras)
plt.grid(True)

# 保存图表
#plt.savefig('result_for_Single-parameter/test_result/r2_scores_bar_chart.png', dpi=300)
plt.clf()