## 说明

- 先对数据进行处理，使用`Lable_encoder.py`将字符型的数据转化为数字类型；
- 接着用`Min-max.py`对数据进行归一化处理；
- `Process_Imbanlce.py`平衡正负样本差异；
- `get_feature.py`筛选特征，为**卡方+随机森林**，`Get_corr.py`为计算Pearson相关系数；
- `classify.py`进行模型的训练及分类，并绘制2维分类面，筛选出的为两个特征；
- `3d.py`三维绘图；

分类面的绘制只能绘制出2维和3维