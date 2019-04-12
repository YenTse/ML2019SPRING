#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ML2017 hw3 Plot Model
import sys
from sys import argv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
from keras.models import load_model
from keras.utils import plot_model, np_utils
from sklearn.metrics import confusion_matrix
import keras.backend as K
import load_data

CATEGORY = 7
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# ## Load Data

# In[2]:


print("read train data...")
sys.argv[1] = '../myData/hw3/train.csv' 
print('training file:',sys.argv[1])
D_SET = 10
remander_value = 0
X_train , Y_train = load_data.load(sys.argv[1] , 1)
X_train = X_train.reshape(-1, 48, 48, 1)
#### rescale
X_train = X_train / 255.
#### convert class vectors to binary class matrices (one hot encoding vector)
Y_train = np_utils.to_categorical(Y_train, 7)


x_train = []
y_train = []
x_val = []
y_val = []

#### split validation set, hyperParameters: d_set, remander_value
for i, n in enumerate(X_train):
    if i%D_SET == remander_value:
        x_val.append(n)
        y_val.append(Y_train[i])
    else:
        x_train.append(n)
        y_train.append(Y_train[i])
X_val = np.array(x_val)
Y_val = np.array(y_val)
X_train = np.array(x_train)
Y_train = np.array(y_train)



print ('X_val shape : ' , X_val.shape)
print ('Y_val shape : ' , Y_val.shape)


argv[1] = '../myData/hw3/train.csv'
argv[2] = 'save/20190408/model_0.h5'

print("load model...")
model_name = argv[2]
model = load_model(model_name)

label = ["angry", "disgust", "fear", "happy", "sad", "suprise", "neutral"]


# In[5]:



X = X_val
Y = Y_val
print("plot confusion matrix...")
Y_GD = np.argmax(Y, 1)
print("predict train data...")
predict = model.predict(X, verbose=1)
Y_predict = np.argmax(predict, 1)
confmat = confusion_matrix(Y_GD, Y_predict)
print(confmat)

fig, ax = plt.subplots(figsize=(10.5, 10.5))
ax.matshow(confmat, cmap='YlGnBu', alpha=0.8)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center')
plt.xticks(np.arange(CATEGORY), label)
plt.yticks(np.arange(CATEGORY), label)
plt.xlabel('predicted label')        
plt.ylabel('true label')
plt.savefig("ConfusionMatrix.png")
plt.show()



# In[ ]:




