#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: xieyanduo
"""
import csv
import numpy as np
#import pandas as pd
import math
import sys
import random
from numpy.linalg import inv


def ReadTrainData(filename):
    data = [] 
    for i in range(18):
        data.append([])

    with open(filename, 'r', encoding = 'big5') as csvfile:
        text = csv.reader(csvfile, delimiter = ',')
        i = 0
        for row in text:                #讀取DATA, [data]: 18(測項) * 5760( 24(hr)*20(days)*12(monthes) )
            if i != 0:      #第一行為欄位名稱，不計入data
                for j in range(3, 27):      #只有第3~26格有value
                    if row[j] != 'NR':      
                        data[(i-1)%18].append(float(row[j]))  #記得要換成float，不然原本是str, 無法做矩陣相乘dot()
                    else:
                        data[(i-1)%18].append(float(0))
            i = i + 1

    data = np.array(data)
    data, mean_data, var_data = Normalization(data)
    return data, mean_data, var_data

def ReadTestData(filename):
    test_data = []
    for i in range(18):
        test_data.append([])

    with open(filename, 'r', encoding = 'big5') as csvfile:
        text = csv.reader(csvfile, delimiter = ',')
        for n, row in enumerate(text):
            for j in range(2, 11):
                if row[j] != 'NR':
                    test_data[n%18].append(float(row[j]))
                else:
                    test_data[n%18].append(float(0))
    test_data = np.array(test_data)

    return test_data

def Normalization(data):
    mean_data = np.mean(data, axis=1)
    var_data = data.var(1) ** 0.5
    for x, row in enumerate(data):
        for y, i in enumerate(row):
            data[x][y] = (data[x][y] - mean_data[x])/ var_data[x]

    return data, mean_data, var_data 


def GetTrainPairs(data, featureList):
    x = []
    y = []
    temp = 1
    #生成training pairs(x, y)
    for i in range(12):         #一年有12個月           
        for j in range(471):        #一個月有471筆資料, ex:[1~10, 2~11, ..., 471~480]
            x.append([])        # [x]:5652( 12*471 ) * N depends on featureList
            for k in range(18):    #18個測項 
                    for h in range(9): #前九個小時的每個測項當作feagure
                        #[x] = [ [f11, f12, ..., f1n, f21, f22, ..], .. ]; k代表第K個測項; fi(第i個測項),j(第j個小時)
                        for f in range(featureList[k]):
                            temp *= data[k][i*24*20+j+h]
                            x[i*471+j].append(temp)
                        temp = 1
            y.append(data[9][i*24*20+j+9])

    x = np.array(x)
    y = np.array(y)
    return x, y

def GetTestPairs(data, featureList):
    x = []
    temp = 1
    for i in range(int(data.shape[1]/9)):
        x.append([])
        for j in range(18):
            for h in range(9):
                for f in range(featureList[j]):
                    temp *= data[j][i*9+h]
                    x[i].append(temp)
                temp = 1
    x = np.array(x)
    return x

def GetBatch(x, y, sets):
    '''
    切成ｎ個mini batch
    sets: 分成幾等份
    '''
    vali_x = []
    vali_y = []
    size = x.shape[0]
    lst = [i for i in range(size)]
    random.shuffle(lst)
    if size%sets != 0:
        raise ValueError('training pairs must be divided by vali sets!')
    j = 0
    vali_x.append([])
    vali_y.append([])

    for i in lst:
        if j < size/sets:
            j += 1
            vali_x[-1].append(x[i])
            vali_y[-1].append(y[i])
        else:
            vali_x.append([])
            vali_y.append([])
            j = 0

    return vali_x, vali_y



def GD(x, y, w, iterations=100000):
    #這邊記得加入bias，把原本x矩陣最左邊再加入一行
    #np.concatenate((a, b), axis=0or1)
    #axis = 0 代表新增一列 b 至 a 的最後一列
    #axis = 1 代表新增一行 b 至 a 的最後一行
    #w = np.zeros(x.shape[1])   #w = weight, total 18+2(測項)*9(9小時當作一組features)=bias+180=181, 181*1的矩陣 
    x_t = x.transpose()   
    s_grad = np.zeros(x.shape[1])    #np.zeros() is different to the matlab, = [0, 0, 0, ...]一維向量 162*1
    l_rate = 0.1     #learning rate
    alpha = 0.1
    #iterations = 100000      #迭代次數, 可視為調整w(weight)的次數

    for i in range(iterations):
        t_y = np.dot(x, w)
        loss = t_y - y
        cost = np.sum(loss**2) / len(x)
        cost_avg = math.sqrt(cost)      #去看每一次迭代，平均誤差（cost_avg）應該是要越來越小的
        
        grad = np.dot(x_t, loss)    #Adagrad
        s_grad += grad**2
        ada = np.sqrt(s_grad)
        w = w - l_rate * grad / ada

        #if i%1000 == 0:
            #print('iterations: %d | cost_avg: %f ' %(i, cost_avg))  

    return w    

def Validation(x, y):
    pass

if __name__ == '__main__':

    data, mean_data, var_data = ReadTrainData('train.csv')
    print('data shape:'+str(data.shape))
    featureList = [0, 0, 0, 0, 0, 1, 1, 1, 3, 3, 1, 0, 0, 0, 1, 1, 1, 1]
    #featureList = [0, 0, 1, 0, 0, 1, 0, 1, 3, 3, 1, 0, 1, 0, 0, 0, 0, 0]
    #featureList = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    #featureList = [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0]
    #featureList = [4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4]
    x, y = GetTrainPairs(data, featureList)
    print(len(x), len(x[0]))
    #print(len(y))
    x = np.concatenate((np.ones([len(x), 1]), x),axis = 1)     #x.T means transpose of x, len(x.T[0])=12*471

        
    #print(len(vali_x[0]))
    #input()
    w = np.zeros(x.shape[1])   #w = weight, total 18+2(測項)*9(9小時當作一組features)=bias+180=181, 181*1的矩陣 
    '''
    for epochs in range(20):  # epochs
        mini_x, mini_y = GetBatch(x, y, 12)
        total_loss = 0
        for i in range(10):  # training
            w = GD(np.array(mini_x[i]), np.array(mini_y[i]), w, 1000)
        for i in range(10, 12):  # validate
            y_pred = np.dot(mini_x[i], w)
            loss = y_pred - mini_y[i]
            loss = np.sqrt(np.sum(loss ** 2)/len(mini_y))
            total_loss += loss
        print('validation loss='+str(total_loss/2))
        #input()
    '''
    #mini_x, mini_y = GetBatch(x, y, 12)
    n = 1
    mini_x = x[:-n]
    mini_y = y[:-n]

    inv_XX = inv( np.dot(mini_x.T, mini_x) )
    yy = np.dot(inv_XX, mini_x.T)
    w = np.dot(yy, mini_y)

    test_x = x[-n:]
    test_y = y[-n:]
    y_pred = np.dot(test_x, w)
    loss = (y_pred - test_y)*var_data[9]
    loss = np.sqrt(np.sum(loss ** 2)/len(test_y))
    print('feature list:'+str(featureList))
    print('val loss:'+str(loss))



    print('x, y shape: '+str(x.shape)+str(y.shape))
    print('w shape:'+str(w.shape))
    np.save('model.npy', w)
    #print(w)

   
    test_data = ReadTestData('test.csv')
    for x, row in enumerate(test_data):  #Normalize testing data
        for y, i in enumerate(row):
            test_data[x][y] = (test_data[x][y] - mean_data[x])/var_data[x]


    print('test_data shape:'+str(test_data.shape))
    #input()
    test_x = GetTestPairs(test_data, featureList)
    #print('test_x shape:'+str(test_x.shape))
    #adding bias
    test_x = np.concatenate((np.ones([test_x.shape[0], 1]), test_x), axis = 1)
    
    ans = []
    for i in range(test_x.shape[0]):
       ans.append(['id_'+str(i)])
       a = np.dot(w, test_x[i])
       a = a * var_data[9] + mean_data[9]
       ans[i].append(a)
    
    filename = 'predict_ClosedForm.csv'   
    with open(filename, 'w+') as text:
        o = csv.writer(text, delimiter = ',', lineterminator = '\n')
        o.writerow(['id', 'value'])
        for row in ans:
            o.writerow(row)
    print(filename+' saved!')
    


    
   
   



             


       
                
        

    
    
        

        


