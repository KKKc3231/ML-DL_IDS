# -*- coding:utf-8 -*-
# 作者：KKKC
import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D


data = pd.read_csv("C:/Users/HP/Desktop/3_feture.csv")
X = data.iloc[:,0:-1]
Y = data.iloc[:,-1]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1)

model = svm.SVC(kernel='linear',C=1.5)
model.fit(X_train,Y_train)
print('Train score:{}'.format(model.score(X_train,Y_train)))
print('Test score:{}'.format(model.score(X_test,Y_test)))

n_Support_vector = model.n_support_  # 支持向量个数
sv_idx = model.support_  # 支持向量索引
w = model.coef_  # 方向向量W
b = model.intercept_

ax = plt.subplot(111, projection='3d')
x = np.arange(0,1,0.01)
y = np.arange(0,1,0.01)
x, y = np.meshgrid(x, y)

z = (w[0, 0] * x + w[0, 1] * y + b) / (-w[0, 2])
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1)

#  训练样本
x_array = np.array(X_train, dtype=float)
y_array = np.array(Y_train, dtype=int)
# 测试样本
X_array = np.array(X_test,dtype=float)
Y_array = np.array(Y_test,dtype=int)

# 训练集和测试集的两类样本,0是正样本，1为负样本
pos_train = x_array[np.where(y_array == 0)]
neg_train = x_array[np.where(y_array == 1)]

pos_test = X_array[np.where(Y_array == 0)]
neg_test = X_array[np.where(Y_array == 1)]

# ax.scatter(pos_train[:, 0], pos_train[:, 1], pos_train[:, 2], c='r', label='pos_train')
# ax.scatter(neg_train[:, 0], neg_train[:, 1], neg_train[:, 2], c='b', label='neg_train')
ax.scatter(pos_test[:, 0], pos_test[:, 1], pos_test[:, 2], c='g', label='pos_test')
ax.scatter(neg_test[:, 0], neg_test[:, 1], neg_test[:, 2], c='orange',label='neg_test')


# # 支持向量绘制
# X = np.array(X_train,dtype=float)
# for i in range(len(sv_idx)):
#     ax.scatter(X[sv_idx[i],0], X[sv_idx[i],1], X[sv_idx[i],2],s=50,
#                 c='',marker='o', edgecolors='g')

# 坐标轴
ax.set_zlabel('Z')
ax.set_ylabel('Y')
ax.set_xlabel('X')
ax.set_zlim([0, 1])
plt.legend(loc='upper left')
ax.view_init(35,300)
plt.show()

