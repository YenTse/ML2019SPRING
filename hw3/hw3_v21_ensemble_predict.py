#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import sys
import load_data
import glob
import os
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import time


def output(pred_list,text) :    
    out = [['id','label']]

    for i in range(len(pred_list)) :
        tmp = [str(i) , int(pred_list[i])]
        out.append(tmp)
    
    pd.DataFrame(out).to_csv(text , header = False , index = False) 


def predict(dir_path , data) :    
#     model = load_model(model_name)
#     model_aug = load_model('64753.h5')

#     pred_list = model.predict(data)
#     pred_list = pred_list + model_aug.predict(data)
#     pred = np.argmax(pred_list , axis = 1) 
#     file_list = glob.glob(os.path.join(dir_path, '*.h5'))
#     print(file_list)
#     file_list.sort()
#     input()
    
    model = load_model('model_0.h5')
    preds = model.predict(data)
    for i in range(1, 10):
        print(i)
        modelname = 'model_'+str(i)+'.h5'
        model = load_model(modelname)
        preds += model.predict(data)
    pred = np.argmax(preds , axis = 1) 

    return pred

def main() :
#     sys.argv[1] = '../myData/hw3/test.csv'
#     sys.argv[2] = 'ensemble_0to9_20190408.csv'
    X_test = load_data.load(sys.argv[1] , 0)
    X_test = X_test.reshape(-1, 48, 48, 1)
    X_test = X_test / 255

    pred = predict('save' , X_test)
    output(pred.astype('int') , sys.argv[2])
    print('predict done...')

if __name__ == "__main__":
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    main()


# In[ ]:




