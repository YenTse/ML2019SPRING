import csv
import numpy as np
#import pandas as pd
import math
import sys
import random
import pandas as pd
from numpy.linalg import inv
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib

def main(args):

    test_data = pd.read_csv(args[5])
    #print(test_data)

    # load model
    model = joblib.load('gbm.pkl')

    preds = model.predict(test_data)


    ans = []
    for i in range(preds.shape[0]):
       ans.append([str(i+1)])
       ans[i].append(int(preds[i]))

    # save answer to a csv file
    filename = args[6]  
    with open(filename, 'w+') as text:
        o = csv.writer(text, delimiter = ',', lineterminator = '\n')
        o.writerow(['id', 'label'])
        for row in ans:
            o.writerow(row)
    print(filename+' saved!')
    

if __name__ == '__main__':
    main(sys.argv)

