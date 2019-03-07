import csv
import numpy as np
#import pandas as pd
import math
import sys
import random
from numpy.linalg import inv

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

def main(args):

    test_data = ReadTestData(args[1])
    #print(test_data)
    
    # initial parameters: mean_data, var_data, featureList
    _init = []
    with open('init_parameters.csv', 'r', encoding = 'big5') as csvfile:
        text = csv.reader(csvfile, delimiter = ',')
        for n, row in enumerate(text):
            _init.append([])
            for j in row:
                _init[-1].append(float(j))
    mean_data = _init[0]
    var_data = _init[1]
    featureList = [int(i) for i in _init[2]]


    # Normalize testing data
    for x, row in enumerate(test_data):  
        for y, i in enumerate(row):
            test_data[x][y] = (test_data[x][y] - mean_data[x])/var_data[x]


    test_x = GetTestPairs(test_data, featureList)       
    test_x = np.concatenate((np.ones([test_x.shape[0], 1]), test_x), axis = 1)  #adding bias

    # Load weight
    w = np.load('model.npy')

    ans = []
    for i in range(test_x.shape[0]):
       ans.append(['id_'+str(i)])
       a = np.dot(w, test_x[i])
       a = a * var_data[9] + mean_data[9]
       ans[i].append(a)
    
    filename = args[2]   
    with open(filename, 'w+') as text:
        o = csv.writer(text, delimiter = ',', lineterminator = '\n')
        o.writerow(['id', 'value'])
        for row in ans:
            o.writerow(row)
    print(filename+' saved!')
    print(ans)


    

if __name__ == '__main__':
    main(sys.argv)

