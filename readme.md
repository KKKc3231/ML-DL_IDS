# 🤖ML&DL——IDS入侵检测

## 1、说明

- 本文档记录机器学习作业，实现IDS入侵检测，数据集使用的UNSW_NB15公开的数据集，采用的方法参考论文《基于深度神经网络的网络入侵检测技术》根据论文的思路实现了机器学习和深度学习的方法来进行入侵检测。

- 深度学习方法使用了简单的CNN、LSTM网络进行分类，没有使用论文中的BiGRU网络

- 提出自己的一些改进方法，主要是在提取特征阶段，由于该数据集中特征属性是不连续的，不太适合使用皮尔逊相关系数来筛选特征，初步的思想是可以使用卡方过滤搭配RF来进行特征选择。

- 对不平衡类别样本的处理，可以使用adasyn算法

- 环境pip install -r requirements.txt 即可，注意keras和tensorflow的对应关系

  [tensorflow和keras对应关系](https://blog.csdn.net/weixin_40109345/article/details/106730050)
  
  

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

`ML/Get_corr.py`

计算不同特征之间的相关性系数，与RF一同进行特征选择，这个方法可以试一试，我使用的是**Chi + RF**，不过效果好像并没有太大提升。

### 2.5 RFP

`RFP.py`

RF和Pearson特征选择

### 2.6 卡方过滤和RF

​	详见文档`IDS.md`

## 3、可视化

svm可视化

2维特征和3维特征的可视化绘图。

删除特征`attack_cat`，因为该特征为类别特征，效果和`label`的效果一样。

2d：`classify.py`

3d：`3d.py`

绘图的具体可参考博客：

[SVM 分类器的分类超平面的绘制（2d）](https://blog.csdn.net/ericcchen/article/details/79332781?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-2.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-2.pc_relevant_default&utm_relevant_index=5)

[SVM分类器绘制3维分类超平面](https://blog.csdn.net/u011995719/article/details/81157193)

[SVM简介及sklearn参数](https://www.cnblogs.com/solong1989/p/9620170.html)

<img src="https://gitee.com/kkkcstx/kkkcs/raw/master/img/20220412110717.png" style="zoom:67%;" />



## 4、CNN

`cnn_train.py`

使用最基础的cnn进行二分类，框架使用tensorflow的keras，简单方便，也可以使用pytorch，训练集`Train-min-max.csv`，是进行归一化后的数据。

需要注意的是，`categorical_crossentropy`损失函数对应的标签为**独热编码**。

`cnn_predict.py`

加载训练好的模型，在测试集上测试，测试集`Test-min-max.csv`

## 5、LSTM

长短期记忆网络，相对于CNN关注更多的时序信息。

模型的搭建与CNN类似，见`lstm_train.py`和`lstm_predict.py`



