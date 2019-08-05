# -*- coding: UTF-8 -*-


import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.datasets import mnist
from struct import unpack

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000,784) # 将图片摊平，变成向量
x_test = x_test.reshape(10000,784) # 对测试集进行同样的处理
x_train = x_train / 255
x_test = x_test / 255
y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)
model = Sequential()
model.add(Dense(input_dim = 28*28,units = 500,activation='relu'))
model.add(Dense(units = 500,activation='relu'))
model.add(Dense(units = 10,activation='softmax'))
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'Adam',metrics = ['accuracy'])
model.fit(x_train,y_train,batch_size = 100,nb_epoch = 12)
score = model.evaluate(x_test,y_test)
print("loss:",score[0])
print("accu:",score[1])