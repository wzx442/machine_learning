"""
Created on 2024/5/11 18:19
@author: 王中校
"""
import numpy as np
from utils.features import prepare_for_training  # 预处理


class LinearRegression:
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        # 初始化函数
        """
        1.对数据进行预处理操作
        2.先得到所有的特征个数
        3.初始化参数矩阵
        """
        (data_processed,  # 预处理之后的数据
         features_mean,  # mean值
         features_devitation  # 标准差
         ) = prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True)
        self.data = data_processed  # 使用预处理之后的数据进行后续操作
        self.labels = labels
        self.feature_mean = features_mean
        self.feature_devitation = features_devitation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        num_features = self.data.shape[1]  # shape[1]表示有多少个列
        self.theta = np.zeros((num_features, 1))

    def train(self, alpha, num_iterations=500):  # alpha是学习率；num_iterations是迭代次数
        """
        训练模块，执行梯度下降
        """
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history

    def gradient_descent(self, alpha, num_iterations):
        """批量梯度下降中的实际迭代模块，会迭代num_iterations次"""
        cost_history = []  # 每一次损失的变化
        for _ in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data, self.labels))
        return cost_history

    def gradient_step(self, alpha):
        """
        梯度下降参数更新方法
        注意是矩阵运算
        """
        num_examples = self.data.shape[0]  # data.shape[0]是样本个数
        prediction = LinearRegression.hypothesis(self.data, self.theta)  # 预测值
        delta = prediction - self.labels  # 预测值-真实值  残差
        theta = self.theta
        theta = theta - alpha * (1 / num_examples) * (np.dot(delta.T, self.data)).T
        self.theta = theta

    def cost_function(self, data, labels):
        """损失函数，损失计算方法"""
        num_example = data.shape[0]  # 样本个数
        delta = LinearRegression.hypothesis(self.data, self.theta) - labels  # 当前的预测值
        cost = (1 / 2) * np.dot(delta.T, delta)/num_example
        return cost[0][0]

    @staticmethod  # 静态方法 调用静态方法时，不需要创建类实例，只需要使用类名即可
    def hypothesis(data, theta):
        """np.dot()是向量的点积或内积"""
        """np.dot()会得到一个行向量"""
        prediction = np.dot(data, theta)  # 用data 乘 theta
        return prediction

    def get_cost(self,data,labels):
        data_processed = prepare_for_training(data,
         self.polynomial_degree,
         self.sinusoid_degree,
         self.normalize_data
         )[0]

        return self.cost_function(data_processed,labels) #损失值

    def prdict(self,data):
        """用训练好的参数模型，预测回归结果"""
        data_processed = prepare_for_training(data,
                                              self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data
                                              )[0]
        predictions = LinearRegression.hypothesis(data_processed,self.theta)
        return predictions
