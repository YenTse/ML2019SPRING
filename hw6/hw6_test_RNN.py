#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import jieba
from gensim.models import word2vec
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Embedding, CuDNNGRU, Dense, TimeDistributed, BatchNormalization, LeakyReLU, Dropout, LSTM, GRU, Bidirectional
import sys
import pandas as pd 
import os


# ## hyper-parameters

# In[2]:


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# TRAIN_X_PATH = '../myData/hw6_data/train_x.csv'
# TRAIN_Y_PATH = '../myData/hw6_data/train_y.csv'

TEST_X_PATH = sys.argv[1]
SAVE_PATH = sys.argv[3]
DICT_TXT_BIG_PATH = sys.argv[2]
# MODEL_PATH = 'model_kaggle_best_0758.h5'
MODEL_PATH = 'model_kaggle_best.h5'




# In[3]:


print('start')
# load traditional chinese dictionary
jieba.load_userdict(DICT_TXT_BIG_PATH)
# load word to vector model
w2v_model = word2vec.Word2Vec.load("dcard_word2vec.model")


#embedding layer
embedding_matrix = np.zeros((len(w2v_model.wv.vocab.items()) + 1, w2v_model.vector_size))
word2idx = {}

vocab_list = [(word, w2v_model.wv[word]) for word, _ in w2v_model.wv.vocab.items()]
for i, vocab in enumerate(vocab_list):
    word, vec = vocab
    embedding_matrix[i + 1] = vec
    word2idx[word] = i + 1


embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                            output_dim=embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            trainable=False
                            )


def text_to_index(corpus):
    new_corpus = []
    for doc in corpus:
        new_doc = []
        for word in doc:
            try:
                new_doc.append(word2idx[word])
            except:
                new_doc.append(0)
        new_corpus.append(new_doc)
    return np.array(new_corpus)

test_data = pd.read_csv(TEST_X_PATH)
test_x = test_data['comment'].values
# train_y = pd.read_csv(TRAIN_Y_PATH)['label'].values
seg_test_x = []
for row in test_x:
    sega = jieba.cut(row, cut_all=False)
    sega = ','.join(sega).split(',')
    seg_test_x.append(sega)


print('seg_test_x', seg_test_x[0])

test_data = text_to_index(seg_test_x)


"""
#normalize?
mean = np.mean(train_data)
std = np.std(train_data)
train_data = (train_data-mean)/std
"""

#padding
padding_length = 200
test_data = pad_sequences(test_data, maxlen=padding_length)


model = Sequential()
model = load_model(MODEL_PATH)
prediction = model.predict_classes(test_data)
# print(prediction[0])

# save prediction
print('predicting...')
with open(SAVE_PATH, 'w') as f:
    print('id,label', file = f)
    for i in range(prediction.shape[0]):
        print('%d,%d' % (i,prediction[i][0]), file = f)

print(SAVE_PATH+'saved...')


# In[ ]:





# In[ ]:




