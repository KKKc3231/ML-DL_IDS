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
    Convolution1D(3,64,padding='same',activation='relu',input_shape=(42,1)),
    Convolution1D(64,128,padding='same',activation='relu'),
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