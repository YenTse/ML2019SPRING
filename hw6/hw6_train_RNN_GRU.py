#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
np.random.seed(8+9)
import jieba
from gensim.models import word2vec
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, CuDNNGRU, Dense, TimeDistributed, BatchNormalization, LeakyReLU, Dropout, LSTM, GRU, Bidirectional
from keras.callbacks import ModelCheckpoint
import sys
import pandas as pd 
import os
## delete plt when upload
#import matplotlib.pyplot as plt


# ## hyper-parameters

# In[2]:


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
# config = tf.ConfigProto()  
# config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
# session = tf.Session(config=config)


# TRAIN_X_PATH = '../myData/hw6_data/train_x.csv'
TRAIN_X_PATH = sys.argv[1]
# TRAIN_Y_PATH = '../myData/hw6_data/train_y.csv'
TRAIN_Y_PATH = sys.argv[2]
# DICT_TXT_BIG_PATH = './dict.txt.big'
DICT_TXT_BIG_PATH = sys.argv[4]
SAVE_MODEL_NAME = 'model_kaggle_best.h5'
BTACH_SIZE = 512


# ## build model

# In[3]:


def RNN(embedding_layer):
    #RNN model
    model = Sequential()
    model.add(embedding_layer)
    model.add(Bidirectional(GRU(256, return_sequences=True)))
    model.add(Bidirectional(GRU(256, return_sequences=True)))
    #model.add(Dense(256, activation='relu'))
    model.add(TimeDistributed(Dense(256, activation='relu')))
    model.add(BatchNormalization())
    #model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    #model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation = 'softmax'))

    model.summary()

    return model


# In[4]:


## load traditional chinese word library
jieba.load_userdict(DICT_TXT_BIG_PATH)
## load word to vector model
w2v_model = word2vec.Word2Vec.load("dcard_word2vec.model")
print('w2v_model: ', len(w2v_model.wv.vocab.items()))
print('w2v_model vector size: ', w2v_model.vector_size) # 每個字詞用幾維的vector表示
vector = w2v_model.wv['喜歡']
# print(vector)

#embedding layer
embedding_matrix = np.zeros((len(w2v_model.wv.vocab.items()) + 1, w2v_model.vector_size)) # (29483, 250)

word2idx = {}

vocab_list = [(word, w2v_model.wv[word]) for word, _ in w2v_model.wv.vocab.items()]  # [ (word, vector), .... ]
# print('vocab_list: ', vocab_list[0])
# input()

# 建立一個字典，w2v_model中的字詞對應到一個整數，並存入(word2idx)，
# 這個整數可以在embedding_matrix中找到字詞的向量
for i, vocab in enumerate(vocab_list):
    word, vec = vocab
    embedding_matrix[i + 1] = vec
    word2idx[word] = i + 1


embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                            output_dim=embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            trainable=False
                            )

# keras Embedding 要求輸入為整數編碼，所以先把每筆data由分割過的string轉為整數編號
def text2index(corpus):
    new_corpus = []
    for doc in corpus:
#         print('doc: ', doc)
        new_doc = []
        for word in doc:
            try:
                new_doc.append(word2idx[word])
            except:
                new_doc.append(0)
#         print('new_doc: ', new_doc)
#         input()
        new_corpus.append(new_doc)
    return np.array(new_corpus)

#cut the sentence
# input_datas = [list(jieba.cut(i.split(',')[1])) for i in open(sys.argv[1], 'r').read().split('\n')[1:-1]]
train_data = pd.read_csv(TRAIN_X_PATH)
train_x = train_data['comment'].values
train_y = pd.read_csv(TRAIN_Y_PATH)['label'].values
seg_train_x = []
for row in train_x:
    sega = jieba.cut(row, cut_all=False)
    sega = ','.join(sega).split(',')
    seg_train_x.append(sega)

#load and generate train data/label
print(len(train_y))
label = to_categorical(np.array(train_y)).reshape(len(train_y),1,2)
# print('label: ', label)
# input()
train_data = text2index(seg_train_x)


"""
#normalize?
mean = np.mean(train_data)
std = np.std(train_data)
train_data = (train_data-mean)/std
"""

# padding Sequences 
# keras只能處理相同長度的序列，所以要先對每個序列進行預處理，使長度一致
padding_length = 200
train_data = pad_sequences(train_data, maxlen=padding_length)

model = RNN(embedding_layer)


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_crossentropy', 'accuracy'])

#earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint(SAVE_MODEL_NAME, save_best_only=True, monitor='val_acc', mode='max', verbose=1, save_weights_only=False)


history = model.fit(x=train_data, y=label, batch_size=BTACH_SIZE, epochs=50, validation_split=0.1, callbacks=[mcp_save])

# plot 
#plt.clf()
#plt.plot(model.history.history['acc'])
#plt.plot(model.history.history['val_acc'])
#plt.title('Training Process_RNN')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['acc', 'val_acc'], loc='upper left')
#plt.savefig(SAVE_MODEL_NAME[:-3] + "_his.png")



import pickle
with open('./TrainingHistoryDict', 'wb') as f:
    pickle.dump(history.history, f)




# In[ ]:




