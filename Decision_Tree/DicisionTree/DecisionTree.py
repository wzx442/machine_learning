"""
Created on 2024/5/13 11:01
@author: 王中校
"""
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator


def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['F1-AGE', 'F2-WORK', 'F3-HOME', 'F4-LOAN']
    return dataSet, labels


def createTree(dataset, labels, featureLabels):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):  # 节点数等于叶子数
        return classList
    if len(dataset[0]) == 1:
        return majorityCnt(classList)
    bestFeature = chooseBestFeature(dataset)  # 获得最佳标签
    bestFeatureLabel = labels[bestFeature]  # 得到最佳标签
    featureLabels.append(bestFeatureLabel)  # 将最佳标签加入特征标签集合中
    myTree = {bestFeatureLabel: {}}  # 构建树模型
    del labels[bestFeature]  # 删除选中的标签
    featureValue = [example[bestFeature] for example in dataset]
    uniqueValues = set(featureValue)
    for value in uniqueValues:
        subLabels = labels[:]
        myTree[bestFeatureLabel][value] = createTree(splitDataset(dataset, bestFeature, value), subLabels,
                                                     featureLabels)
    return myTree


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    SortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reversed=True)
    return SortedClassCount[0][0]

def chooseBestFeature(dataset):
    numFeatures = len(dataset[0]) - 1  # 计算特征数量
    bestEntropy = calcShannonEnt(dataset)  # 熵值

def calcShannonEnt(dataset):  # 计算熵值
    numexample = len(dataset)  # 标签总数
    labelCount = {}
    for featureVec in dataset:
        currentLabel = featureVec[-1]
        if currentLabel not in labelCount.keys():  # 当前标签不在标签集合中
            labelCount[currentLabel] = 0  # 置为0
        labelCount[currentLabel] += 1

    shannonEnt = 0
    for key in labelCount:  # 计算熵值
        prop = float(labelCount[key])/numexample
        shannonEnt -= prop*log(prop, 2)
    return shannonEnt