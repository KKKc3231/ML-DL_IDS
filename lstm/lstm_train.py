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
# X_train = np.array(X_train)
# Y_train = np.array(Y_train)
# data = pd.DataFrame(X_train)
# Y_train = to_categorical(Y_train) # 转换为独热编码 one-hot
print(Y_train.shape)
x_train = np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1])) # 将数据集转化为（行数，特征个数，1），input_shape（特征，1）
print(x_train)

# lstm网络参数
number_lstm = 72
lstm = Sequential([
    LSTM(units=number_lstm,activation='relu',input_shape=(1,42),return_sequences=True),
    Dropout(0.5),
    Dense(1),
    Activation('sigmoid'),
])
# 对于binary_crossentropy损失函数，y不可以用独热编码
lstm.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="results/lstm_results/checkpoint-{epoch:02d}.hdf5",
                                         verbose=1,save_best_only=True,monitor='val_acc',mode='max')
# csvlogger = CSVLogger('results/cnn2results/cnntrainanalysis2.csv',separator=',', append=False)
#
lstm.fit(x=x_train,y=Y_train,batch_size=128,epochs=5,callbacks=[checkpointer])
lstm.save(filepath='results/lstm_results/lstm_model.hdf5')