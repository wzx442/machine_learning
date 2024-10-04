"""
Created on 2024/10/4 14:34
@author: 王中校
"""
import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms     # 用来数据转换
import torch.utils.data as data_utils
from CNN import CNN


############# 数据加载 ###################

train_data = dataset.MNIST(
    root="mnist",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_data = dataset.MNIST(
    root="mnist",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

# print(train_data)
# print(test_data)

# 用DataLoader分批加载
train_loader = data_utils.DataLoader(dataset=train_data,
                                     batch_size=64, # 一批次多少条
                                     shuffle=True,  #打乱数据,避免过拟合
                                     )

test_loader = data_utils.DataLoader(dataset=test_data,
                                     batch_size=64, # 一批次多少条
                                     shuffle=True,  #打乱数据,避免过拟合
                                     )

# print(train_loader)
# print(test_loader)
cnn = CNN()
# cnn = cnn.cuda()
# --------------损失函数-----------------
loss_function = torch.nn.CrossEntropyLoss()   # 交叉熵损失

# -------- 优化函数-----------------
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)

# ---------------训练过程-------------
for epoch in range(10):
    for index, (images, labels) in enumerate(train_loader):
        # print(index)
        # print(images)
        # print(labels)   # 每次都不一样，labels就是数字，因为shuffle为true，所以每次输出都不一样
        # images = images.cuda()
        # labels = labels.cuda()
        outputs = cnn(images)  # 前向传播

        loss = loss_function(outputs, labels)  # 将outputs与labels进行对比

        # 先清空梯度
        optimizer.zero_grad()

        # 反向传播
        loss.backward()
        optimizer.step()

        print("当前为第{}轮, 当前批次为{}/{}, loss为{}".format(epoch+1, index+1, len(train_data) // 64, loss.item()))


    # ----------测试集验证------------
    loss_test = 0   # 总损失
    rightValue = 0
    for index2,(images, labels) in enumerate(test_loader):
        outputs = cnn(images)
        # print(outputs)
        # print(labels)
        loss_test += loss_function(outputs, labels)

        _,pred = outputs.max(1)
        # print(pred)
        # print((pred == labels).sum().item())
        rightValue += (pred==labels).sum().item()
        print("当前为第{}轮测试集验证,当前批次为{}/{},loss为{},准确率为{}".format(epoch+1, index2+1, len(test_data)//64, loss_test,rightValue/len(test_data)))

torch.save(cnn,"model/mnist_model.pkl")




