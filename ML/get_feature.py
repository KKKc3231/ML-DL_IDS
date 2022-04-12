# -*- coding:utf-8 -*-
# 作者：KKKC

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF # random forest
from sklearn.feature_selection import chi2, VarianceThreshold


def Get_feature():
    train_data = pd.read_csv('G:/ML/balanced_data.csv')
    test_data = pd.read_csv('G:/ML/Test-min-max.csv')
    x_train = train_data.iloc[:,0:-1]
    y_train = train_data.iloc[:,-1]
    name = x_train.columns.values
    print(name)
    x_test = test_data.iloc[:,0:-1]
    y_test = test_data.iloc[:,-1]
    # 划分训练集和测试集
    # x_train , x_test , y_train, y_test = train_test_split(X,Y,test_size=0.3)
    chi = chi2(x_train,y_train)
    sort = np.argsort(chi[0])[::-1]# 对卡方值进行排序
    feature = sort[0:30] # 排名前30的特征
    print(feature)

    # name list
    feature_name = []
    for i in feature:
        feature_name.append(name[i])

    print(feature_name)

    # 转化为pd
    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)

    for i in x_train.columns:
        if i not in feature_name:
            x_train.drop(i,axis=1,inplace=True)
            x_test.drop(i,axis=1,inplace=True)

    print(x_train.shape)

    # 构建rf模型进行进一步的特征选择

    model = RF(n_estimators=100,oob_score=True,random_state=1234)
    model = model.fit(x_train,y_train)
    feature_list = model.feature_importances_ # 重要性列表
    feature_sort = np.argsort(feature_list)[::-1] # 排序
    feature_rf = feature_sort[0:2] # 2个特征，筛选几个改几个

    # 获取相应特征的名字
    name = []
    for i in feature_rf:
        name.append(x_train.columns[i])

    print(name)

    # 删除其他特征
    for i in x_train.columns:
        if i not in name:
            x_train.drop(i,axis=1,inplace=True)
            x_test.drop(i,axis=1,inplace=True)

    #
    x_train.to_csv('C:/Users/HP/Desktop/x_train.csv')
    x_test.to_csv('C:/Users/HP/Desktop/x_test.csv')
    y_train.to_csv('C:/Users/HP/Desktop/y_train.csv')
    y_test.to_csv('C:/Users/HP/Desktop/y_test.csv')
    return x_train,x_test,y_train,y_test
