import time
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score



x, y = pickle.load(open('data/train.pkl','rb'))




x = np.concatenate(x)
y = np.concatenate(y)

'''
x = x[:1000]
y = y[:1000]
'''

train_x, test_x , train_y, test_y = train_test_split(x,y,test_size=0.1)


print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
print(test_y)


model = SVC(gamma=0.0008)

print("fitting model")

before = time.time()

model = model.fit(train_x, train_y)

after = time.time()


print('fitted model:',  float(after-before) / 60.0 , "minutes")

print("predicting")

before = time.time()
pred = model.predict(test_x)
print(pred)

precision = precision_score(y_true=test_y, y_pred=pred)
recall = recall_score(y_true=test_y,y_pred=pred)
f1 = f1_score(test_y,pred)

print("precision:", precision, "recall:" , recall , "f1:" , f1)

scores = model.decision_function(test_x)

roc = roc_auc_score(test_y,scores)

print("ROC score:" ,roc)

after = time.time()

print("scoring took " , float(after - before) / 60.0 , 'minutes')

pickle.dump(model,open('models/model_init.pkl', 'wb'))


