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
print(x_train)

# 搭建cnn模型
cnn = Sequential([
    Convolution1D(3,64,padding='same',activation='relu',input_shape=(42,1)), # input_shape-> 42个特征，一次读一行
    Convolution1D(64,128,padding='same',activation='relu'),
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

cnn.fit(x=x_train,y=Y_train,batch_size=256,epochs=5,callbacks=[checkpointer])
cnn.save(filepath='results/cnn2results/cnn_model.hdf5')