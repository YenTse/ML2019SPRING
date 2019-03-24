import numpy as np
import pandas as pd 
import csv
import sys
import random
from numpy.linalg import inv
def Read_Data(filepath):
    df = pd.read_csv(filepath)
    o = np.array(df)
    return o

class GenerativeModel():
    def __init__(self):
        pass
    def init_parameters(self, xshape):
        #self.w = np.zeros((xlength+1, 1))
        self.x_len, self.feature_len = xshape
        pass
    def normalization(self, data, isTrain=0):
        if isTrain:
            self.mean_ = np.mean(data, axis=0)
            self.std_ = np.std(data, axis=0)

        return (data - self.mean_) / (self.std_+1e-20)
    def add_bias(self, data, data_len):
        return np.concatenate((np.ones((data_len, 1)), data), axis=1)
    def sigmoid(self, x):
        return np.array(1 / (1 + np.exp(-1 * x)))
    def cross_entropy(self, labels, predicts):
        a = np.dot(np.transpose(labels), np.log(predicts))
        b = np.dot(np.transpose((np.ones((len(labels), 1))-labels)), np.log(np.ones((len(labels), 1))-predicts))
        return -1 * (a + b) 
    def predict(self, x):
    
        px_c1 = np.exp(-1/2*np.dot(np.dot(x-self.mu1, inv(self.share_covmatrix)), (x-self.mu1).T))
        #print(x.shape)
        #input()
        px_c2 = np.exp(-1/2*np.dot(np.dot(x-self.mu2, inv(self.share_covmatrix)), (x-self.mu2).T))
        a = (px_c1 * self.class1_len/self.x_len)
        b = (px_c2 * self.class2_len/self.x_len)
        predict = a / (a + b)
        #print(predict)
        return predict 
    def gd(self, x, y, pred):
        return -np.dot(x.T, (y - pred))
    def fit(self, x, y, class_num, batch_size=0, lr=0.5):
        self.init_parameters(x.shape)
        x = self.normalization(x, isTrain=1)
        #x = self.add_bias(x, x.shape[0])
        #x_class1 = [x[i] for i in range(y.shape[0]) if y[i]==1]
        x_class = []
        # seperate data into different class
        for i in range(class_num):
            x_class.append([])
        for i, row_data in enumerate(x):
            x_class[int(y[i])].append(row_data)

        #print(len(x_class[0]))
        #input()
        self.class1_len = len(x_class[0])
        self.class2_len = len(x_class[1])
        self.mu1 = np.mean(x_class[0], axis=0)
        self.mu2 = np.mean(x_class[1], axis=0)
        self.covar_maxtrix1 = 1/len(x_class[0]) * np.dot((x_class[0] - self.mu1).T, (x_class[0] - self.mu1))
        self.covar_maxtrix2 = 1/len(x_class[1]) * np.dot((x_class[1] - self.mu2).T, (x_class[1] - self.mu2))
        #print(self.covar_maxtrix2.shape)
        #input()
        self.share_covmatrix = (len(x_class[0])/len(x)) * self.covar_maxtrix1 + (1 - len(x_class[0])/len(x)) * self.covar_maxtrix2
        


def main(args):
    x = Read_Data(args[3])
    y = Read_Data(args[4])
    x_test = Read_Data(args[5])
    class_num = 2  # 2-class problem
    #print(y)
    #input()

    #data prepocessing
    #####
    #####

    model = GenerativeModel()
    model.fit(x, y, class_num)
    #print(model.w)

    
    ### Testing data #######
    x_test = model.normalization(x_test, isTrain=0)
    #x_test = model.add_bias(x_test, x_test.shape[0])
    #preds = model.sigmoid(model.predict(x_test))
    temp = []
    for i, value in enumerate(x_test):
        #print(value)
        preds = model.predict(value)  # value.shape=(106, )
        if preds > 0.5:
            temp.append(0)
        else:
            temp.append(1)


    ans = []
    for i in range(len(temp)):
       ans.append([str(i+1)])
       ans[i].append(int(temp[i]))

    # save answer to a csv file
    filename = args[6]   
    with open(filename, 'w+') as text:
        o = csv.writer(text, delimiter = ',', lineterminator = '\n')
        o.writerow(['id', 'label'])
        for row in ans:
            o.writerow(row)
    print(filename+' saved!')

    #print(x.shape)

    




if __name__ == '__main__':
    main(sys.argv)
