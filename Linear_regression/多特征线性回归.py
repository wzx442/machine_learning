"""
Created on 2024/5/12 20:28
@author: 王中校
"""
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

plotly.offline.init_notebook_mode()

data = pd.read_csv('../data/world-happiness-report-2017.csv')  # 读取数据

# 得到训练和测试数据
train_data = data.sample(frac=0.8)          # 80%的值用于训练
test_data = data.drop(train_data.index)     # 剩下的值用于测试

input_param_name1 = 'Economy..GDP.per.Capita.'
input_param_name2 = 'Freedom'
output_param_name = 'Happiness.Score'

x_train = train_data[[input_param_name1, input_param_name2]].values
y_train = train_data[[output_param_name]].values

x_test = test_data[input_param_name1, input_param_name2].values
y_test = test_data[output_param_name].values

plt.scatter(x_train, y_train, label='train_data')  # 画出数据图
plt.scatter(x_test, y_test, label='test_data')     # 画出散点图
plt.xlabel(input_param_name1, input_param_name2)                       # x轴名称
plt.ylabel(output_param_name)                      # y轴名称
plt.title('Happy')
plt.legend()
plt.show()

num_iteratons = 500  # 迭代次数
learning_rate = 0.01  # 学习率

linear_regression = LinearRegression(x_train, y_train)  # 初始化
(theta, cost_history) = linear_regression.train(learning_rate, num_iteratons)  # 训练

print('开始时的损失: ', cost_history[0])
print('训练后的损失: ', cost_history[-1])

plt.plot(range(num_iteratons), cost_history)
plt.xlabel('迭代次数')
plt.ylabel('损失')
plt.title('梯度下降')
plt.show()


preditions_num = 100
x_predictions = np.linspace(x_train.min(), x_train.max(), preditions_num).reshape(preditions_num, 1)  # 100*1的矩阵
y_predictions = linear_regression.prdict(x_predictions)  # 得到预测结果值

plt.scatter(x_train, y_train, label='Train_data')
plt.scatter(x_test, y_test, label='Test_data')
plt.plot(x_predictions, y_predictions, 'r', label='预测值')
plt.xlabel(input_param_name1,)                       # x轴名称
plt.ylabel(output_param_name)                      # y轴名称
plt.title('happy')
plt.legend()
plt.show()
