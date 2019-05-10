#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jieba
import pandas as pd
from gensim.models import Word2Vec


# ## hyper-parameters

# In[2]:


TRAIN_X_PATH = '../myData/hw6_data/train_x.csv'
TRAIN_Y_PATH = '../myData/hw6_data/train_y.csv'


# ## read data

# In[3]:


train_data = pd.read_csv(TRAIN_X_PATH)
train_x = train_data['comment'].values
train_y = pd.read_csv(TRAIN_Y_PATH)['label'].values
print(train_x)


# ## word segmentation (jieba)

# In[4]:


'''
@jieba tutorial: https://github.com/fxsjy/jieba
'''
seg_list = jieba.cut('人生短短幾個秋', cut_all=False)
print("Default Mode: " ,  ','.join(seg_list).split(','))

seg_train_x = []
for row in train_x:
    sega = jieba.cut(row, cut_all=False)
    sega = ','.join(sega).split(',')
    seg_train_x.append(sega)
print(len(seg_train_x))


# In[ ]:





# ## word embedding (gensim)

# In[6]:


'''
@ gensim tutorial: https://radimrehurek.com/gensim/models/word2vec.html
'''
from gensim.models import word2vec
model = word2vec.Word2Vec(seg_train_x, size=250, window=5, min_count=5, workers=4, iter=10, sg=1)

model.save("dcard_word2vec.model")


# In[ ]:





# In[ ]:





# In[ ]:




