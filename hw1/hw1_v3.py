#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 14:18:01 2018

@author: xieyanduo
"""
import csv
import numpy as np
import math
import sys
import pandas as pd 
import random

def changeFeatureList(featureList):
    columes = ['AMB_TEMP', 'CH4', 'CO', 'NHMC', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 
    'PM2.5', 'RAINFALL', 'RH', 'SO2', 'THC', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR']
    ans = []
    for i in featureList:
        ans.append(columes.index(i))
    return ans 

#def shuffleData(x):

def getRawData(filename):
    
        
    df = pd.read_csv(filename, delimiter=',', encoding='big5')
    df = df.fillna(0)  # fill NA => 0
    df.replace('NR', 0, inplace=True)  # replace all 'NR' to 0
    #data size: 18*5760
    df = df.iloc[:, 3:]  # delete index, or columes name
    
    df = np.array(df)

    data =[]
    for i in range(18):
        data.append([])

    for i, row in enumerate(df):
        for k in row:
            data[i%18].append(float(k))

    return data
    

def getTrainData(data, featureList, isValidRatio = 0):
    
    X = []
    Y = []
    #generate training pairs(x, y)
    for i in range(12):         #12 months per year           
        for j in range(471):        #471 data per month, ex:[1~10, 2~11, ..., 471~480]
            X.append([])        # [x]:5652( 12*471 ) * 162( features = 18*9 )
            for k in range(18):    #18 testing items 
                if k not in featureList: # select feature, 8:PM10, 9:PM2.5
                    continue
                for m in range(9): #前九個小時的每個測項當作feagure
                    #[x] = [ [f11, f12, ..., f1n, f21, f22, ..], .. ]; k代表第K個測項; fi(第i個測項),j(第j個小時)
                    X[i*471+j].append(data[k][i*24*20+j+m])
            Y.append(data[9][i*24*20+j+9])

    X = np.array(X)
    Y = np.array(Y)
    randList = [i for i in range(len(Y))]
    new_X = []
    new_Y = []
    random.shuffle(randList)
    for i in range(len(Y)):
        new_X.append(X[randList[i]])
        new_Y.append(Y[randList[i]])

    if isValidRatio != 0:
        n = int(len(Y)/(isValidRatio+1)*isValidRatio)
        X = new_X[:n]
        Y = new_Y[:n]
        validX = new_X[n+1:]
        validY = new_Y[n+1:]
        return X, Y, validX, validY
    else:
        return new_X, new_Y
    
    

    




    

class LinearRegression():

    def __init__(self):
        pass  

    def parameter_init(self, dim):
        self.w = np.zeros((dim, 1))
        self.b = 0

    def feature_scaling(self, X, train=False):    
        if train:
            self.min = np.min(X, axis=0)
            self.max = np.max(X, axis=0)
        return (X - self.min) / (self.max - self.min)

    def train(self, X, Y, validX, validY, epochs=10000, batch_size=100, lr=0.00000000000001): # valid_X, valid_Y
        
        self.parameter_init(dim=len(X[0]))
        print(self.b)
        for i in range(epochs):
            randList = [i for i in range(len(Y))]
            random.shuffle(randList)
            s_grad = np.zeros((len(X[0])+1, 1))
            for batch in range(int(len(Y)/batch_size)):  # set batch x, y
                x = []  # x: batch_size * ( len(featureList)*9 )
                y = []  # y: batch_size * 1
                for k in range(batch_size):
                    temp = randList[batch*batch_size+k]
                    x.append([k for k in X[temp]])
                    y.append([Y[temp]])
                x = np.array(x)
                y = np.array(y)
                y_pred = np.dot(x, self.w) + self.b
                temp = np.dot(x, self.w)
                loss = y_pred - y
                w_gred = np.dot(x.T, loss)
                b_gred = sum(loss)
                self.w = self.w - w_gred * lr 
                self.b = self.b - b_gred * lr 
            #print('loss:' + str(sum(loss)/len(y)))
            #print(self.b.shape)
            if i%100 == 0:
                print('epochs:'+str(i)+', loss:'+str(np.sqrt(sum(loss**2)/batch_size)))
    def test(self, filename):
        pass
        




def main(args):
    train_filename = args[1]
    test_filename = args[2]
    predict_filename = args[3]
    featureList = ['PM2.5']

    featureList=changeFeatureList(featureList)
    df = getRawData(train_filename)
    X, Y, validX, validY = getTrainData(df, featureList, isValidRatio = 3)

    hw1 = LinearRegression()
    #hw1.parameter_init(len(X[0]))
    #print(len(X[0]))
    hw1.train(X, Y, validX, validY)





if __name__ == '__main__':
    main(sys.argv)



             


       
                
        

    
    
        

        


