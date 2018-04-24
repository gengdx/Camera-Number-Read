# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 09:47:48 2018

@author: GengDx
"""
import numpy as np
import os
import scipy
import cv2
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, InputLayer
from keras.optimizers import Adam
from keras.optimizers import SGD

(X_train, y_train),(X_test,y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28,1).astype('float32')  
X_test = X_test.reshape(-1,28, 28,1).astype('float32')  
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)
X_train=X_train/255.
X_test=X_test/255.

#当以上mnist.load_data()不起作用时是因为天朝Great Wall起作用，采用以下代码或网络上自行搜索方案
#from tensorflow.examples.tutorials.mnist import input_data  
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  
#   
#X_train, y_train = mnist.train.images,mnist.train.labels  
#X_test, y_test = mnist.test.images, mnist.train.labels  
#X_train = X_train.reshape(-1, 28, 28,1).astype('float32')  
#X_test = X_test.reshape(-1,28, 28,1).astype('float32')  
#y_train = np_utils.to_categorical(y_train,num_classes=10)
#y_test = np_utils.to_categorical(y_test,num_classes=10)


# Another way to build your CNN
model=Sequential()
#model.add(InputLayer(input_shape=(28,28,1)))

# Conv layer 1 output shape
model.add(Convolution2D(
        nb_filter=32,
        kernel_size=(5,5),
        padding='same',
        input_shape=(28,28,1),
        ))
model.add(Activation('relu'))

#Pooling layer 1 (max pooling) output shape (32,60,60)
model.add(MaxPooling2D(
        pool_size=(2,2),
        strides=(2,2),
        padding='same',
        ))

#Conv layer 2 output shape (64,60,60)
model.add(Convolution2D(64,kernel_size=(5,5),padding='same'))
model.add(Activation('relu'))

#Pooling layer2 output shape(64,12,12)
model.add(MaxPooling2D(
        pool_size=(2,2),
        strides=(2,2),
        padding='same',
        ))

#Fully connected layer 1 input shape (64*12*12)=9216, output shape(1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
#Fully connected layer 2 to shape(2)
model.add(Dense(10))
model.add(Activation('softmax'))

adam=Adam(lr=1e-4)
sgd=SGD(lr=0.05, decay=1e-6,momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training -------------')
model.fit(X_train,y_train,epochs=20,batch_size=128,)

print('\nTesting-------------')
loss, accuracy=model.evaluate(X_test,y_test)
print('\ntest loss:',loss)
print('\ntest accuracy:',accuracy)

model.save('G:/CNN Model/Number_NET.h5')    #存储神经网络模型
model.summary()
