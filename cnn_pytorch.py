# -*- coding: utf-8 -*-
# @Time : 2023/3/23 10:30
# @Author : KKKc
# @FileName: cnn_pytorch.py
# pytorch版本的实现，对原始数据进行训练并预测

import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from loguru import logger # 输出日志的库
import pandas as pd
import numpy as np

# 超参数
epoch = 10000
learning_rate = 1e-4

# iris数据，用来测试模型
# iris = load_iris()
# x = iris.data
# y = iris.target

data = pd.read_csv("G:/ML/Train-min-max.csv") # 你自己的数据集，最后一列是特征，前面列是数据
x = data.iloc[:,0:-1]
y = data.iloc[:,-1]
x = np.array(x) # dataframe -> numpy
y = np.array(y)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3) # 划分数据集
 
# model
class my_cnn(nn.Module):
    def __init__(self,num_feat=42):
        super(my_cnn, self).__init__()
        self.l1 = nn.Linear(num_feat,16)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(16,3) # 最后输出的3对应的3个类别

    def forward(self,x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x


net = my_cnn(num_feat=42) # num_feat是你数据集的输出特征，如果是iris的花改成4，根据你特征选择后的数据集进行修改即可
optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate) # 优化器
loss_f = nn.CrossEntropyLoss() # 交叉熵损失函数

# numpy转tensor
x_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(Y_train)

x_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(Y_test)

for i in range(epoch):
    out = net(x_train)
    loss = loss_f(out,y_train)
    logger.info("The {} epoch loss: {}".format(i+1,loss))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
torch.save(net.state_dict(),'cnn_model_{}.pth'.format(epoch)) # 保存模型
out = net(x_test) # 网络输出
pre = torch.max(out,1)[1] # 最大的值对应的下标即为得到的类别
pre = pre.data.numpy()
y_test = y_test.data.numpy()
acc = accuracy_score(pre,y_test)
print("test acc:{}".format(acc))
