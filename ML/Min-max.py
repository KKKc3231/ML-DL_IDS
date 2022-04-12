# -*- coding:utf-8 -*-
# 作者：KKKC

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

X = pd.read_csv('G:/ML/Test_Process_data.csv')
columns = X.columns.values.tolist()
Min_max = MinMaxScaler().fit_transform(X) # 归一化
new_X = pd.DataFrame(Min_max,columns=columns)
print(new_X)
new_X.to_csv('G:/ML/Test-min-max.csv',index=None)