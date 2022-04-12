# -*- coding:utf-8 -*-
# 作者：KKKC

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF # random forest
from sklearn.metrics import accuracy_score
from sklearn import svm # svm 模型
from sklearn.linear_model import LogisticRegression as LR # logistic
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
from ml import Get_corr

# 读取训练集数据
data = pd.read_csv('G:/ML/balanced_data.csv')
X = data.iloc[:,0:-1]
Y = data.iloc[:,-1]
# X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3) # 划分训练集和测试集
X_train = X
Y_train = Y
# 构建RF模型进行特征选择
model1 = RF(n_estimators=100, oob_score=True,random_state=1234)
model1 = model1.fit(X_train, Y_train) # 训练RF模型
feture = model1.feature_importances_ # 得出特征重要性
print('feture:') # print以下
# print(feture)
feture_sort = np.argsort(feture)[::-1]
# 取前15维度的特征
feture_rf = feture_sort[0:15]
print(feture_rf)

# 获取相应特征的名字
name = []
for i in feture_rf:
    name.append(X.columns[i])

# RFP确定删除的特征，
delete_feture = ['ct_dst_src_ltm','ct_srv_src']
new_name = []

# 删去高相关性且特征重要性低的特征
for i in name:
    if i not in delete_feture:
        new_name.append(i)
print(new_name)

# 读取测试集数据
Test = pd.read_csv('G:/ML/Test-min-max.csv')
X_test = Test.iloc[:,0:-1]
Y_test = Test.iloc[:,-1]


# 如果不是RFB选出的特征，则删除，训练集和测试集同时删除
for i in X.columns:
    if i not in new_name:
        X_train.drop(i,axis=1,inplace=True) # 按列删除
        X_test.drop(i,axis=1,inplace=True)

print(np.array(X_test.columns))
print(X_train)
print("Max_feture index:",np.argmax(feture.tolist())) # 最大特征的位置，在原始的数据集中


# 构建模型 -> RF,SVM,KNN,Logistic
model_rf = RF(n_estimators=100, random_state=1234, oob_score=True)
model_svm = svm.SVC()
model_knn = KNN()
model_LR = LR()

# 不同模型的训练
model_svm.fit(X_train,Y_train)
model_knn.fit(X_train,Y_train)
model_LR.fit(X_train,Y_train)
model_rf.fit(X_train, Y_train)

# 不同模型进行预测
Y_pre_rf = model_rf.predict(X_test)
Y_pre_svm = model_svm.predict(X_test)
Y_pre_knn = model_knn.predict(X_test)
Y_pre_lr = model_LR.predict(X_test)

# 不同模型的acc
# 1.RF
score_rf  = accuracy_score(Y_test, Y_pre_rf) # accuracy_score:计算acc
print("-----1、RF-----")
print("True:",np.array(Y_test))
print("Pre:", Y_pre_rf)
print('score_rf:{}'.format(score_rf))
print()

# 2.SVM
score_svm = accuracy_score(Y_test,Y_pre_svm)
print("-----2、SVM-----")
print("True:",np.array(Y_test))
print("Pre:", Y_pre_svm)
print('score_svm:{}'.format(score_svm))
print()

# 3.KNN
score_knn = accuracy_score(Y_test,Y_pre_knn)
print("-----3、KNN-----")
print("True:",np.array(Y_test))
print("Pre:", Y_pre_knn)
print('score_knn:{}'.format(score_knn))
print()

# 4.LR
score_lr = accuracy_score(Y_test,Y_pre_lr)
print("-----4、LR-----")
print("True:",np.array(Y_test))
print("Pre:", Y_pre_lr)
print('score_lr:{}'.format(score_lr))
print()

