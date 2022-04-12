# 机器学习——IDS入侵检测

## 1、说明

- 本文档记录机器学习作业，实现IDS入侵检测，数据集使用的UNSW_NB15公开的数据集，采用的方法参考论文《基于深度神经网络的网络入侵检测技术》根据论文的思路实现了机器学习和深度学习的方法来进行入侵检测。

- 深度学习方法使用了简单的CNN网络进行分类，没有使用论文中的BiGRU网络

- 提出自己的一些改进方法，主要是在提取特征阶段，由于该数据集中特征属性是不连续的，不太适合使用皮尔逊相关系数来筛选特征，初步的思想是可以使用卡方过滤搭配RF来进行特征选择。

- 懂得了如何对不平衡类别样本的处理，可以使用adasyn算法

- 代码结构

  ![image-20220326104958651](https://gitee.com/kkkcstx/kkkcs/raw/master/img/image-20220326104958651.png)

  

## 2、处理过程

### 2.1 字符型特征 --> 数值型特征

`Lable-encoder.py` 

先用drop_duplicates删除重复值后，获取所有的协议。然后使用LabelEncoder()类将字符型特征转化为数值型。

```python
！以proto字段为例子
# 提取proto的字段值，并将其转化为元组中的对应下表
proto = data_copy["proto"].drop_duplicates(inplace=False)
proto = np.array(proto)
print(proto)
# 转换proto
enc0 = preprocessing.LabelEncoder()
enc0 = enc0.fit(proto)
data_copy['proto'] = enc0.transform(data_copy["proto"])
```

### 2.2 归一化

`Min-max.py`

加快模型的训练

```python
Min_max = MinMaxScaler().fit_transform(X) # 归一化
new_X = pd.DataFrame(Min_max,columns=columns)
```

### 2.3 处理不平衡类别样本

`Process_Imbalance.py`

使用adasyn算法进行处理。具体算法流程可参考[ADASYN : 针对不平衡学习的自适应合成抽样方法](https://blog.csdn.net/weixin_50005008/article/details/115178529)。

```python
# adasyn方法平衡正负样本的个数
ada = ADASYN(random_state=1234)
X_res , Y_res = ada.fit_resample(X,Y)
print(X_res)
# 转化为pd的格式
X = pd.DataFrame(X_res,columns=x_columns)
Y = pd.DataFrame(Y_res)
```

### 2.4 特征间相关性系数

`Get_corr.py`

计算不同特征之间的相关性系数，与RF一同进行特征选择

```python
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
    return max_p # 返回相关性列表
```

### 2.5 RFP

`RFP.py`

RF和Pearson特征选择

分类器

```python
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
feture_sort = np.argsort(feture)[::-1]
# 取前15维度的特征
feture_rf = feture_sort[0:2]
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
```

<img src="https://gitee.com/kkkcstx/kkkcs/raw/master/img/屏幕截图 2022-03-22 112841.png" style="zoom:50%;" />

### 2.6 卡方过滤和RF

`Chi_RF.py`

## 3、可视化

svm可视化

2维特征和3维特征的可视化绘图。

删除特征`attack_cat`，因为该特征为类别特征，效果和`label`的效果一样。

<img src="G:\ML\入侵检测\image\2维.png" alt="2维" style="zoom: 67%;" />

`3d.py`

绘图的具体可参考博客：

[SVM 分类器的分类超平面的绘制（2d）](https://blog.csdn.net/ericcchen/article/details/79332781?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-2.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-2.pc_relevant_default&utm_relevant_index=5)

[SVM分类器绘制3维分类超平面](https://blog.csdn.net/u011995719/article/details/81157193)

[SVM简介及sklearn参数](https://www.cnblogs.com/solong1989/p/9620170.html)

```python
# 3d绘图
import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv("C:/Users/HP/Desktop/3_feture.csv")
X = data.iloc[:,0:-1]
Y = data.iloc[:,-1]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)

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

ax.scatter(pos_train[:, 0], pos_train[:, 1], pos_train[:, 2], c='r', label='pos_train')
ax.scatter(neg_train[:, 0], neg_train[:, 1], neg_train[:, 2], c='b', label='neg_train')
ax.scatter(pos_test[:, 0], pos_test[:, 1], pos_test[:, 2], c='g', label='pos_test')
ax.scatter(neg_test[:, 0], neg_test[:, 1], neg_test[:, 2], c='orange',label='neg_test')

# 支持向量绘制
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
```



<img src="https://gitee.com/kkkcstx/kkkcs/raw/master/img/3D.png" alt="svm_3D" style="zoom: 50%;" /><img src="https://gitee.com/kkkcstx/kkkcs/raw/master/img/svm_3D.png" alt="3D" style="zoom: 50%;" />

## 4、CNN

`cnn_train.py`

使用最基础的cnn进行二分类，框架使用tensorflow的keras，简单方便，也可以使用pytorch。

```python
# -*- coding:utf-8 -*-
# 作者：KKKC

from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Embedding
from keras.layers import Convolution1D,MaxPooling1D, Flatten
from keras.datasets import imdb
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Convolution1D, Dense, Dropout, Flatten, MaxPooling1D
from keras.utils import np_utils
from keras import callbacks
from keras.layers import LSTM, GRU, SimpleRNN
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import h5py

data = pd.read_csv("G:/ML/Train-min-max.csv")

X_train = data.iloc[:,0:-1]
Y_train = data.iloc[:,-1]
normal = Normalizer()
X_train = normal.fit_transform(X_train)
Y_train = np.array(Y_train)
# data = pd.DataFrame(X_train)
Y_train = to_categorical(Y_train) # 转换为独热编码 one-hot

x_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
print(x_train.shape)

# 搭建cnn模型
cnn = Sequential([
    Convolution1D(64,3,padding='same',activation='relu',input_shape=(43,1)),
    Convolution1D(64,3,padding='same',activation='relu'),
    MaxPooling1D(),
    Flatten(),
    Dense(128,activation='relu'),
    Dropout(rate=0.5),
    Dense(2,activation='softmax')
])

cnn.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# 定义callbacks
checkpointer = callbacks.ModelCheckpoint(filepath="results/cnn2results/checkpoint-{epoch:02d}.hdf5",
                                         verbose=1,save_best_only=True,monitor='val_acc',mode='max')
# csvlogger = CSVLogger('results/cnn2results/cnntrainanalysis2.csv',separator=',', append=False)

cnn.fit(x=x_train,y=Y_train,batch_size=128,epochs=5,callbacks=[checkpointer])
cnn.save(filepath='results/cnn2results/cnn_model.hdf5')
```

`cnn_predict.py`

加载训练好的模型，在测试集上测试

```python
# -*- coding:utf-8 -*-
# 作者：KKKC
import keras
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Embedding
from keras.layers import Convolution1D,MaxPooling1D, Flatten
from keras.datasets import imdb
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Convolution1D, Dense, Dropout, Flatten, MaxPooling1D
from keras.utils import np_utils
from keras import callbacks
from keras.layers import LSTM, GRU, SimpleRNN
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import h5py

normal = Normalizer()
data = pd.read_csv("G:/ML/Test-min-max.csv")
X_test = data.iloc[:,0:-1]
X_test = normal.fit_transform(X_test)
Y_test = data.iloc[:,-1]
Y_test = np.array(Y_test)
y_test = Y_test
# 独热编码
Y_test = to_categorical(Y_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

cnn = Sequential([
    Convolution1D(64,3,padding='same',activation='relu',input_shape=(43,1)),
    Convolution1D(64,3,padding='same',activation='relu'),
    MaxPooling1D(),
    Flatten(),
    Dense(128,activation='relu'),
    Dropout(rate=0.5),
    Dense(2,activation='softmax')
])

# 加载模型weight
cnn.load_weights("results/cnn2results/cnn_model.hdf5")
y_pre = cnn.predict_classes(x=X_test)

# 保存信息
np.savetxt('res/True.txt',Y_test,fmt="%01d")
np.savetxt('res/Pre.txt',y_pre,fmt="%01d")
cnn.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
loss,accuracy = cnn.evaluate(X_test,Y_test)
# acc = accuracy_score(Y_test,y_pre)
print("loss:{},acc:{}".format(loss,accuracy))
# print("acc:{}".format(acc))
acc = accuracy_score(y_test,y_pre)
print("acc:{}".format(acc))
```

## 5、LSTM

长短期记忆网络

