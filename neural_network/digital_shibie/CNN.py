"""
Created on 2024/10/4 14:54
@author: 王中校
"""
import torch


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = torch.nn.Sequential(
            # 1.卷积
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            # 2. 归一化
            torch.nn.BatchNorm2d(num_features=32),
            # 3.激活函数
            torch.nn.ReLU(),
            # 4. 最大池化
            torch.nn.MaxPool2d(2)
        )
        self.fc = torch.nn.Linear(in_features=14 * 14 * 32, out_features=10)  # 全连接

    def forward(self, x):
        out = self.conv(x)  # 卷积
        out = out.view(out.size()[0], -1)  # 将图像数据展开成一维的
        out = self.fc(out)  # 全连接
        return out
