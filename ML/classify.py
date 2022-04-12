# -*- coding:utf-8 -*-
# 作者：KKKC

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import svm

# 绘制超平面
def plot_hyperplane(clf, X, y,
                    h=0.02,
                    draw_sv=True,
                    title='hyperplan'):
    # 扩大一下范围
    x_min, x_max = X[:, 0].min()-0.1, X[:, 0].max()+0.15
    # x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    y_min, y_max = X[:, 1].min()-0.1, X[:, 1].max()+0.15
    # meshgrid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    plt.title(title)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    # plt.xticks(())
    # plt.yticks(())
    plt.xlabel('X1')
    plt.ylabel('X2')
    # arr = np.concatenate((xx, yy), axis=1)
    print(np.c_[xx.ravel(), yy.ravel()])  # 平面上所有的点
    # SVM的分割超平面
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # 对所有的点进行预测
    # Z = clf.predict(arr)
    print(yy.shape)
    # 绘制等高线
    Z = Z.reshape(xx.shape)
    # print(Z.shape)
    print('---')
    print(xx)
    print('---')
    print(yy)
    print('---')
    print(Z)

    # 绘制等高线的理解
    '''
    xx每一行都相同，每一列不同
    yy每一行都不同，每一列相同
    可以理解为，xx的每一行为x坐标，对于一行yy，对应一个y，如果xx每一行有n个元素，则对于一行yy，则可以产生n个点，
    Z的一行则为这些点对应的类别，即两类等高线。
    '''

    plt.contourf(xx, yy, Z, cmap='bone', alpha=0.2)  # point
    # 绘制等高线，1为一类，0为一类

    # 定义maker和color组合
    markers = ['o', 's']
    colors = ['b', 'r']
    labels = np.unique(y) # 去除重复值，得到[0,1] list
    for label in labels:
        # 绘制两类点
        plt.scatter(X[y==label][:, 0],
                    X[y==label][:, 1],
                    c=colors[label],
                    marker=markers[label])

    # 画出支持向量
    if draw_sv:
        sv = clf.support_vectors_ # 所有的支持向量
        # 绘制支持向量的点，此例中支持向量均为2维
        plt.scatter(sv[:, 0], sv[:, 1], c='y', marker='.',facecolors='none')
    """
    # 绘制test的点
    test_data = pd.read_csv('C:/Users/HP/Desktop/x_test.csv')
    X = test_data.iloc[:, 0:-1]
    y = test_data.iloc[:, -1]
    X = np.array(X)
    y = np.array(y)
    score = clf.score(X,y)
    print(score)
    # 定义maker和color组合
    markers = ['*', '^']
    colors = ['g', 'orange']
    labels = np.unique(y)  # 去除重复值，得到[0,1] list
    for label in labels:
        # 绘制两类点
        plt.scatter(X[y == label][:, 0],
                    X[y == label][:, 1],
                    c=colors[label],
                    marker=markers[label])
    """
    plt.show()

# 生成一个有两个特征、包含两种类别的数据集
# X, y = make_moons(n_samples=1000, centers=2,random_state=0, cluster_std=0.3)
data = pd.read_csv("C:/Users/HP/Desktop/2_feture.csv") # 把特征和标签写道一个csv里面了
XX = data.iloc[:,0:-1]
YY = data.iloc[:,-1]
# X, y = make_moons(n_samples=200,noise=0.05)
clf = svm.SVC(C=2, kernel='rbf') # 可以更改模型，比如rf,knn
clf.fit(XX, YY)
score = clf.score(XX,YY)
print(score)
XX = np.array(XX)
YY = np.array(YY)
# 绘制超平面
plot_hyperplane(clf, XX, YY, h=0.02,title='SVM-rbf')