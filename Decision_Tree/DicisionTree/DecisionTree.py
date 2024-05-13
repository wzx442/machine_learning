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
    baseEntropy = calcShannonEnt(dataset)  # 基础信息增益
    bestInfoGain = 0  # 最好的信息增益
    bestFeature = -1  # 最好的特征
    for i in range(numFeatures):
        featList = [example[i] for example in dataset]
        uniqueValues = set(featList)
        newEntropy = 0  # 新熵值
        for val in uniqueValues:
            subDataset = splitDataset(dataset, i, val)
            prob = len(subDataset) / float(len(dataset))  # 概率值
            newEntropy += prob * calcShannonEnt(subDataset)
        infoGain = baseEntropy - newEntropy  # 信息增益
        if (infoGain > bestInfoGain):  # 更好的信息增益
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def splitDataset(dataset, axis, val):
    retDataSet = []
    for feaVec in dataset:
        if feaVec[axis] == val:
            reducedFeatVec = feaVec[:axis]
            reducedFeatVec.extend(feaVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


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
        prop = float(labelCount[key]) / numexample
        shannonEnt -= prop * log(prop, 2)
    return shannonEnt


'''画图模块'''


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")
    leafNode = dict(boxstyle="round4", fc="0.8")
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = next(iter(myTree))
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')  # 创建fig
    fig.clf()  # 清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # 去掉x、y轴
    plotTree.totalW = float(getNumLeafs(inTree))  # 获取决策树叶结点数目
    plotTree.totalD = float(getTreeDepth(inTree))  # 获取决策树层数
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0  # x偏移
    plotTree(inTree, (0.5, 1.0), '')  # 绘制决策树
    plt.show()


if __name__ == '__main__':
    dataset, labels = createDataSet()
    featLabel = []
    myTree = createTree(dataset, labels, featLabel)
    createPlot(myTree)