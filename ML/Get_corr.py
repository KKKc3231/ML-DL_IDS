# -*- coding:utf-8 -*-
# 作者：KKKC

import pandas as pd

def Get_CorrList():
    data = pd.read_csv('G:/ML/Train-min-max.csv')
    X = data.iloc[:,0:-1]
    columns = X.columns.values.tolist() # 得到标题
    max_p = []
    for i in range(len(columns)):
        print('---.',columns[i])
        a = dict()
        for k in range(len(columns)):
            str = columns[i]+'&'+columns[k]
            if i!=k:
                p = X[columns[i]].corr(X[columns[k]]) # 计算与其他列之间的相关性，不计算对角线
                a[str] = p
        max_key = max(a,key=a.get) # 得出相关性最大的key
        value = a[max_key] # 最大的值
        if value > 0.9: # 限定条件
            print(max_key)
            print(value)
            strlist = max_key.split('&')
            # print(strlist)
            max_p.append(strlist) # 得到所有相关性高于0.9的组合
    # a = []
    # for i in max_p:
    #     for j in i:
    #         a.append(j)
    #
    # print(a)
    print(max_p)
    return max_p # 返回相关性列表