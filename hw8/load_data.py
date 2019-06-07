import numpy as np
import pandas as pd
import time

def load(text  , train = 1):
    file = pd.read_csv(text).values
    data = []
    for i in range(file.shape[0]) :
        data.extend(file[i,1].split())

    data = np.array(data).reshape(file.shape[0],2304).astype('float')
    
    if train == 1 :
        target = file[:,0]
        print ('train loaded done...')
        return  data , target

    print ('test loaded done...')
    return  data


def main():
    X_test = load('test.csv' , 0)
    X_train , Y_train = load('train.csv' , 1)

    pd.DataFrame(X_test.astype('int')).to_csv('X_test_1' , header = None, index = False) 
    pd.DataFrame(X_train.astype('int')).to_csv('X_train_1' , header = None, index = False) 
    pd.DataFrame(Y_train.astype('int')).to_csv('Y_train_1' , header = None, index = False) 
    
if __name__ == "__main__":
    main()


