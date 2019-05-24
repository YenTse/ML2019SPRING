#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys
import os
from skimage.io import imread
from skimage.io import imsave
import glob


# ## prepare dataset

# In[2]:


def prepare_dataset(data_path):
    x_train = []
    img_list = glob.glob(os.path.join(data_path, '*.jpg'))
#     print(img_list)
    for img_path in img_list:
        img = imread(img_path)
        img = img.flatten().astype('float32') 
#         print(img.shape)
#         print(img)
#         input()
        x_train.append(img)
    x_train = np.array(x_train)
    
    return x_train


# ## define PCA

# In[3]:


class PCA():
    def __init__(self):
        self.shape = (600, 600, 3)

    def Mean_face(self, data, save_filename, isSave=0):
        mean = np.mean(data, 0)
        if isSave:
            #print(self.shape)           
            imsave(save_filename, mean.reshape(self.shape).astype(np.uint8))
            print('< {} saved >\n'.format(save_filename))

    def Eigen_face(self, data, isSave=0):
        mean = np.mean(data, 0)
        x = (data - mean)
        print('calcuating svd...')
#         U, S, V = self.U, self.S, self.V
        U, S, V = np.linalg.svd(x.T, full_matrices=False) 
        print(U.shape)
        S_sum = np.sum(S)
        for i in range(5):
            print(S[i]/S_sum)
#         print(self.U.shape)
#         input()
#         print(self.U.shape)
#         input()
        for i in range(0, 10):
            eigen = U[:, i]
            if isSave:
                eigen -= np.min(eigen)
                eigen /= np.max(eigen) 
                eigen = (255*eigen).astype(np.uint8)
#                 print(eigen.shape)
#                 input()
                save_name = 'eigen_face_' + str(i) + '.jpg'
#                 print(np.transpose(eigen, (2, 0, 1)).shape)
#                 print(eigen.T.shape)
#                 input()
                imsave(save_name, eigen.reshape(self.shape).astype(np.uint8))
#                 io.imsave(os.path.join(args.output_path, save_name), eigen.T.astype(np.uint8))
                print('< {} saved >\n'.format(save_name))

    def Reconstruct(self, data, x_target, eigenNum, isSave, save_name):
        print('Reconstruct...')
        mean = np.mean(data, 0)
        x = (data - mean)          
#         U, S, V = self.U, self.S, self.V
        U, S, V = np.linalg.svd(x.T, full_matrices=False)
    
#         self.U, self.S, self.V = np.linalg.svd(x.T) 
        x_target_center = x_target - mean
        weight = np.dot(x_target_center, U[:, :eigenNum])
        recog = mean + np.dot(weight, U[:, :eigenNum].T)
        recog -= np.min(recog)
        recog /= np.max(recog) * 1
        recog = (255*recog).astype(np.uint8)
        if isSave:
            imsave(save_name, recog.reshape(self.shape).astype(np.uint8))
            print('< {} saved >\n'.format(save_name))
        return recog


# In[4]:


# data_path = '../../myData/hw7_data/Aberdeen/'
data_path = sys.argv[1]
input_img = sys.argv[2]
save_name = sys.argv[3]
x_train = prepare_dataset(data_path)
# print(x_train.shape)  # x_train.shape = (415, 600, 600, 3)
input_img_id = int(input_img[:-4])

# In[5]:


pca = PCA()
pca.Reconstruct(x_train, x_train[input_img_id], 5, isSave=1, save_name = save_name)


# pca.Mean_face(x_train, 'meanface.jpg', isSave=1)
# pca.Eigen_face(x_train, isSave=1)

# rec = [10, 20, 30, 40, 50]
# for i in rec:
#     pca.Reconstruct(x_train, x_train[i], 5, isSave=1, save_name = str(i) + '_reconstruct' + '.jpg')

# In[ ]:





# In[ ]:




