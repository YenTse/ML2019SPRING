#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sys

from keras.models import load_model
import keras.backend as K

from sklearn.cluster import KMeans
import glob
import os
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# sys.argv[1] = '../myData/hw7_data/images/'
# train = np.load(sys.argv[1])
# test = pd.read_csv(sys.argv[2]).as_matrix()
# train = train.astype('float32') / 255.

img_dir = './../myData/hw7_data/images/'
# img_list = glob.glob(os.path.join(img_dir, '*.jpg'))
train = []
# for img_path in img_list:
# #     print(img_path)
#     img = Image.open(img_path).convert('L')
#     img = np.array(img)
#     img = img.reshape(32*32)
#     train.append(img)

for i in range(1, 40001):
    name = '0000000'+str(i)
    name = name[-6:] + '.jpg'
#     print(name)
#     input()
    img = Image.open(os.path.join(img_dir, name)).convert('L')
    img = np.array(img)
    img = img.reshape(32*32)
    train.append(img)
    
    
# train = np.load('data/image.npy')
train = np.array(train)
print (train.shape)
train = train.astype('float32') / 255.


# In[ ]:


print('load testing data')

test = pd.read_csv('./../myData/hw7_data/test_case.csv').values

print(test)
input()

def output(pred_list,text) :    
    out = [['id','label']]

    for i in range(len(pred_list)) :
        tmp = [str(i) , pred_list[i]]
        out.append(tmp)
    
    pd.DataFrame(out).to_csv(text , header = False , index = False) 

autoencoder = load_model('./model/best_model_95047.h5')

# results = autoencoder([train[0]])
# print(results)
# input()


autoencoder.summary()
encoder = K.function([autoencoder.get_layer('input_1').input, autoencoder.get_layer('dense_1').input], [autoencoder.get_layer('dense_2').output])

pred = encoder([train])
pred = np.array(pred)

print(pred.shape) ##pred.shape=(1, 40000, 384)
# input()

# from sklearn.manifold import TSNE
# tsne = TSNE(n_components=24, verbose=1, perplexity=50, n_iter=3000)
# X_2d = tsne.fit_transform(pred[0])


from MulticoreTSNE import MulticoreTSNE as TSNE
X_2d = TSNE(n_jobs=12, random_state=17, n_iter=3000).fit_transform(pred[0])

labels = X_2d


kmeans = KMeans(n_clusters=2, random_state=0).fit(labels)
labels = kmeans.labels_


print('len of label: ', len(labels))
print('label: ', labels)


# input()

# print (sum(labels))
# A = [0, 1, 2, 3, 6, 8, 9, 11, 13, 14, 16, 17]    # for best_model_8246
# A = [5, 18, 17, 0, 12, 10, 15, 3, 8, 13, 15, 2]  #for best_model_90
# A = [1, 11, 0, 15, 4, 18, 17, 13, 12, 8, 16, 10]  #for best_model_95047
# A = [8, 3, 22, 14, 20, 21, 23, 12, 2, 6, 18, 13, 15, 0] # acc: 95504
A = [0] # for TSNE_2

out = []
for i in range(1000000) :
    if labels[test[i][1]-1] in A:
        a = 0
    else:
        a = 1
    if labels[test[i][2]-1] in A:
        b = 0
    else:
        b = 1
    if a == b:
        out.append(1)
    else:
        out.append(0)
    
#     if labels[test[i][1]-1] == labels[test[i][2]-1] :
#         out.append(1)
#     else : 
#         out.append(0)
out = np.array(out)
print(out.shape)
# input()

output(out , 'predict.csv')
print('predict saved...')

# for i, n in enumerate(labels):
#     name = '0000000'+str(i+1)
#     name = name[-6:] + '.jpg'
#     print('image: ', i+1, 'label: ', n)
#     im = Image.open(os.path.join('../myData/hw7_data/images/', name))
#     im.show()
#     input()



# In[ ]:



# autoencoder = load_model('best.h5')
# autoencoder.summary()


# In[3]:


# A = [7, 0, 2, 4, 8]   #for tsne:2, kmeans:11
# out = []
# for i in range(1000000) :
#     if labels[test[i][1]-1] in A:
#         a = 0
#     else:
#         a = 1
#     if labels[test[i][2]-1] in A:
#         b = 0
#     else:
#         b = 1
#     if a == b:
#         out.append(1)
#     else:
#         out.append(0)

# out = np.array(out)
# print(out.shape)
# # input()

# output(out , 'predict.csv')
# print('predict saved...')





# In[ ]:




