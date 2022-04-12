# -*- coding:utf-8 -*-
# 作者：KKKC

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


data = pd.read_csv('G:/ML/UNSW_NB15_test-set.csv',low_memory=False) # 读取数据

X = data.iloc[:,:]
Y = data.iloc[:,-1]
print(X)
print(pd.DataFrame(data))
print(data["label"].shape)

# 复制训练数据
data_copy = X

# 提取proto的字段值，并将其转化为元组中的对应下表
proto = data_copy["proto"].drop_duplicates(inplace=False)
proto = np.array(proto)
print(proto)
# print(np.where(proto == 'tcp')[0][0]) # 得到转化后的值

# 提取service的字段值，并将其转化为元组中的对应下表
service = data_copy["service"].drop_duplicates(inplace=False)
service = np.array(service)
print(service)
# print(np.where(service == '-')[0][0])

# 提取state的字段值，并将其转化为元组中的对应下表
state = data_copy["state"].drop_duplicates(inplace=False)
state = np.array(state)
print(state)

'''
# 提取attack_cat的字段值，并将其转化为元组中的对应下表
attack_cat = data_copy["attack_cat"].drop_duplicates(inplace=False)
attack_cat = np.array(attack_cat)
print(attack_cat)
'''

# 转换proto
enc0 = preprocessing.LabelEncoder()
enc0 = enc0.fit(proto)
data_copy['proto'] = enc0.transform(data_copy["proto"])

# 转换service
enc1 = preprocessing.LabelEncoder()
enc1 = enc1.fit(service)
print(data_copy['service'])
data_copy['service'] = enc1.transform(data_copy['service'])
print(data_copy['service'])
data = pd.DataFrame(data_copy['service'])
data.to_csv('G:/ML/1.csv')
# 转换state
enc2 = preprocessing.LabelEncoder()
enc2 = enc2.fit(state)
data_copy['state'] = enc2.transform(data_copy['state'])

'''
# 转换attack_cat
enc3 = preprocessing.LabelEncoder()
enc3 = enc3.fit(attack_cat)
data_copy['attack_cat'] = enc3.transform(data_copy['attack_cat'])
'''

data = pd.DataFrame(data_copy)
data.to_csv('G:/ML/Test_Process_data.csv',index=None)
