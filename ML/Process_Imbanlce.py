# -*- coding:utf-8 -*-
# 作者：KKKC

import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN

data = pd.read_csv('G:/ML/Train-min-max.csv')
X = data.iloc[:,0:-1]
x_columns = X.columns.values.tolist() # 获取标题
Y = data.iloc[:,-1]
# y_columns = Y.columns.values.tolist() # too
print(X)
print(Y)
# adasyn方法平衡正负样本的个数
ada = ADASYN(random_state=1234)
X_res , Y_res = ada.fit_resample(X,Y)
print(X_res)
# 转化为pd的格式
X = pd.DataFrame(X_res,columns=x_columns)
Y = pd.DataFrame(Y_res)

# 存储
X.to_csv('balance_X.csv',index=None)
Y.to_csv('balance_Y.csv',index=None)