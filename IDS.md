# 机器学习+深度学习 实现IDS

​	在老师所给的材料中，使用皮尔逊相关性系数和随机森林进行特征选择，但是由于皮尔逊相关性过滤处理的是连续性数据，但特征中存在离散型数据，可使用卡方检验进行过滤。

​	实现方法：SVM、RF、CNN、LSTM、GRU

## 卡方过滤

<div align=center>
<img src="C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20220407193322505.png"/>
</div>

## 随机森林特征选择

​	对数据进行卡方过滤之后，可紧接着进行rf特征选择，rf根据特征的重要性进行特征选择，特征重要性的原理如下：

在随机森林中某个特征X的重要性的计算方法如下：

​	在[随机森林之Bagging法](http://www.cnblogs.com/justcxtoworld/p/3434057.html)中可以发现Bootstrap每次约有1/3的样本不会出现在Bootstrap所采集的样本集合中,当然也就没有参加决策树的建立,那是不是意味着就没有用了呢,答案是否定的。我们把这1/3的数据称为袋外数据oob（out of bag）,它可以用于取代测试集误差估计方法.

[随机森林之oob error 估计 - 人若无名 - 博客园 (cnblogs.com)](https://www.cnblogs.com/justcxtoworld/p/3434266.html)

- 对于随机森林中的每一颗决策树,使用相应的OOB(袋外数据)数据来计算它的袋外数据误差,记为err1.

- 随机地对袋外数据OOB所有样本的特征X加入噪声干扰(就可以随机的改变样本在特征X处的值),再次计算它的袋外数据误差,记为err2.

- 假设随机森林中有N棵树,那么对于特征X的重要性=∑(err2-err1)/N,之所以可以用这个表达式来作为相应特征的重要性的度量值是因为：**若给某个特征随机加入噪声之后,袋外的准确率大幅度降低,则说明这个特征对于样本的分类结果影响很大,也就是说它的重要程度比较高。**

  原文链接：https://blog.csdn.net/qq_31307013/article/details/80454190

## 进行二分类并绘制分类平面

可使用RF、SVM、KNN、LR模型对筛选出的特征进行分类，并绘制二维分类面，需要使用到np.meshgrid()以及等高线的绘制，具体的细节详见`classify.py`

---

SVM的分类效果图如下：

<img src="https://gitee.com/kkkcstx/kkkcs/raw/master/img/svm二分类.png" style="zoom:67%;" />



