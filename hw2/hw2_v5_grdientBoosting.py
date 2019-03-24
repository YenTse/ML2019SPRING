import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
#from sklearn.grid_search import GridSearchCV
import sys
import csv

args = sys.argv
#read data
X = pd.read_csv(args[3])
y = pd.read_csv(args[4])
test = pd.read_csv(args[5])
#predicts = pd.read_csv(args[4])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=876, test_size=0.30)


gbm = GradientBoostingClassifier(random_state=7678, n_estimators=300)
#gbm0.fit(X_train,y_train)
gbm.fit(X,y)

# save model
model_name = 'gbm.pkl'
joblib.dump(gbm, model_name)
print('{} saved!'.format(model_name))


preds = gbm.predict(test)
#print(preds)


ans = []
for i in range(preds.shape[0]):
   ans.append([str(i+1)])
   ans[i].append(int(preds[i]))
#print(ans)

# save answer to a csv file
'''
filename = args[6]  
with open(filename, 'w+') as text:
    o = csv.writer(text, delimiter = ',', lineterminator = '\n')
    o.writerow(['id', 'label'])
    for row in ans:
        o.writerow(row)
print(filename+' saved!')
'''





