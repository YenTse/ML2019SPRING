#!/usr/bin/env python
# coding: utf-8

# In[7]:


from keras.layers import Conv2D, MaxPooling2D , LeakyReLU , BatchNormalization
from keras.layers import Dense, Dropout, Activation , Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.models import Sequential
from keras.utils import np_utils
from keras import optimizers
import keras.callbacks
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import load_data
import time
import sys
import os
from model import CNN_Model


# In[3]:


def fit(model, X_train, Y_train, epochs = 20, BATCH_SIZE = 256, d_set=10, remander_value=0, SaveModel_name='Untitled.h5') :
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam() , metrics=['accuracy'])
    #### generate data
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)
    datagen.fit(X_train)   # only featurewise_center, featurewise_std_normalization, zca_whitening need to use fit
    print(X_train.shape)
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    
    #### split validation set, hyperParameters: d_set, remander_value
    for i, n in enumerate(X_train):
        if i%d_set == remander_value:
            x_val.append(n)
            y_val.append(Y_train[i])
        else:
            x_train.append(n)
            y_train.append(Y_train[i])
    X_val = np.array(x_val)
    Y_val = np.array(y_val)
    X_train = np.array(x_train)
    Y_train = np.array(y_train)
    print(X_val.shape)

    #earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint(SaveModel_name, save_best_only=True, monitor='val_acc', mode='max', verbose=1, save_weights_only=True)

    history = model.fit_generator(datagen.flow(X_train, Y_train,batch_size=BATCH_SIZE , save_to_dir = None, seed=1314),
                        steps_per_epoch=X_train.shape[0] / BATCH_SIZE,
                        validation_data=(X_val, Y_val),
                        epochs=epochs,
                        verbose=1, 
                        callbacks=[mcp_save], shuffle=True)

    return model


#### load data and reshapeimport sys
# sys.argv[1] = './../myData/hw8_data/train.csv'
print('training file:',sys.argv[1])
D_SET = 10
X_train , Y_train = load_data.load(sys.argv[1] , 1)
X_train = X_train.reshape(-1, 48, 48, 1)

#### rescale
X_train = X_train / 255.

print ('X_train shape : ' , X_train.shape)
print ('Y_train shape : ' , Y_train.shape)

#### convert class vectors to binary class matrices (one hot encoding vector)
Y_train = np_utils.to_categorical(Y_train, 7)


#### build model
for i in range(1):
    print('No.'+str(i))
    SaveModel_name = 'model_best_62998.h5'
    model = CNN_Model()
    #model = fit(model , X_train , Y_train , epochs = 50 , val_split = 0.2)
    model = fit(model, X_train, Y_train, epochs = 400, d_set=D_SET, remander_value=i, SaveModel_name=SaveModel_name)
#         plt.clf()
#         plt.plot(model.history.history['acc'])
#         plt.plot(model.history.history['val_acc'])
#         plt.title('Training Process_CNN')
#         plt.ylabel('accuracy')
#         plt.xlabel('epoch')
#         plt.legend(['acc', 'val_acc'], loc='upper left')
#         plt.savefig(SaveModel_name[:-3] + "_his.png")

    score = model.evaluate(X_train, Y_train, verbose=0)
    print('Train score:', score[0])
    print('Train accuracy:', score[1])




