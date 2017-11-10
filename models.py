#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 21:51:52 2017

@author: kaku
"""


from keras.models import Model
from keras.layers import Input, MaxPooling2D, UpSampling2D
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import keras


def CNN_mnist(patch_size, num_classes):
    """
    From keras example
    """
    ISZ = patch_size    
    inputs = (ISZ, ISZ, 3)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=inputs))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    print('Hello!')
else:
    print('Different models can be loaded.')
    





