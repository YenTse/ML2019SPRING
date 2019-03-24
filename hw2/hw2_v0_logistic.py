import numpy as np
import pandas as pd 
import csv
import sys
import random
def Read_Data(filepath):
    df = pd.read_csv(filepath)
    o = np.array(df)
    return o

class Logistic_Regression():
    def __init__(self):
        pass
    def init_parameters(self, xlength):
        self.w = np.zeros((xlength+1, 1))
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
        return np.dot(x, self.w)
    def gd(self, x, y, pred):
        return -np.dot(x.T, (y - pred))
    def fit(self, x, y, epoch=1000, batch_size=0, lr=0.5):
        self.init_parameters(x.shape[1])
        x = self.normalization(x, isTrain=1)
        x = self.add_bias(x, x.shape[0])
        for i in range(1, 1+epoch):
            pred = self.sigmoid(self.predict(x))
            loss = self.cross_entropy(y, pred)
            #grad = gd(x, y, pred)
            w_grad = np.dot(np.transpose(x), (y - pred)) * (-1/pred.shape[0]) 
            self.w = self.w - w_grad * lr

            if i%1000 == 0:
                print('loss:' + str(loss/y.shape[0]))

def main(args):
    x = Read_Data(args[3])
    y = Read_Data(args[4])
    x_test = Read_Data(args[5])
    #print(y)
    #input()
    #data prepocessing
    #####
    #####

    model = Logistic_Regression()
    model.fit(x, y, epoch=20000)
    print(model.w)

    #testing data
    x_test = model.normalization(x_test, isTrain=0)
    x_test = model.add_bias(x_test, x_test.shape[0])
    preds = model.sigmoid(model.predict(x_test))
    for i, value in enumerate(preds):
        if value > 0.5:
            preds[i] = 1
        else:
            preds[i] = 0
    print(preds)

    ans = []
    for i in range(preds.shape[0]):
       ans.append([str(i+1)])
       ans[i].append(int(preds[i][0]))

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
