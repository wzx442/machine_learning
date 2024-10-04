"""
Created on 2024/10/4 16:29
@author: 王中校
"""
import cv2
import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from CNN import CNN

test_data = dataset.MNIST(
    root="mnist",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

test_loader = data_utils.DataLoader(dataset=test_data,
                                     batch_size=64, # 一批次多少条
                                     shuffle=True,  #打乱数据,避免过拟合
                                     )

# 加载训练好的模型
cnn = torch.load("model/mnist_model.pkl")


# --------------损失函数-----------------
loss_function = torch.nn.CrossEntropyLoss()   # 交叉熵损失


loss_test = 0  # 总损失
rightValue = 0

for index, (images, labels) in enumerate(test_loader):
    outputs = cnn(images)  # 前向传播
    _, pred = outputs.max(1)
    loss_test += loss_function(outputs, labels)
    rightValue += (pred == labels).sum().item()

    images = images.cpu().numpy()     # 将张量转换为numpy格式
    labels = labels.cpu().numpy()
    pred = pred.cpu().numpy()

    for idx in range(images.shape[0]):
        im_data = images[idx]
        im_data = im_data.transpose(1,2,0)
        im_label = labels[idx]
        im_pred = pred[idx]
        print("预测值为{}".format(im_pred))
        print("真实值为{}".format(im_label))
        cv2.imshow(winname="now_Image", mat=im_data)
        cv2.waitKey(0)

print("loss为{},准确率为{}".format(loss_test, rightValue / len(test_data)))