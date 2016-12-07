import time
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import os
from math import exp


dropbox_dir = os.path.expanduser("~/Dropbox/ActiveLearningBackup/")

_lambda = 0.5
_gamma = 0.0008

def main():
	print(dropbox_dir)
	x_init , y_init = pickle.load(open('data/init.pkl','rb'))

	x_query, y_query = pickle.load(open('data/query.pkl','rb'))

	x_test, y_test = pickle.load(open('data/test.pkl','rb'))


	x_init = np.concatenate(x_init)
	y_init = np.concatenate(y_init)

	x_test = np.concatenate(x_test)
	y_test = np.concatenate(y_test)


	'''
	active_learning_step = 0

	model = SVC(kernel='rbf', gamma=_gamma)

	model.fit(x_init[:1000],y_init[:1000])

	evaluate_model(model,active_learning_step,x_test,y_test)

	'''

	print(get_average_cos(x_query))

def evaluate_model(model,learning_step, x_test, y_test):
	pickle.dump(model, open(dropbox_dir+'model_{}.pkl'.format(learning_step),'wb'))
	pred = model.predict(x_test)
	print(pred)

	print("Evaluating model at learning step", learning_step)
	precision = precision_score(y_true=y_test, y_pred=pred)
	recall = recall_score(y_true=y_test,y_pred=pred)
	f1 = f1_score(y_test,pred)



	scores = model.decision_function(x_test)

	roc = roc_auc_score(y_test,scores)

	print("precision:", precision, "recall:" , recall , "f1:" , f1, "roc" , roc)

	eval_f = open(dropbox_dir+'eval_{}'.format(learning_step), 'w')
	eval_f.write('precision,recall,f1,roc\n')
	eval_f.write('{},{},{},{}\n'.format(precision,recall,f1,roc))
	eval_f.close()

def get_query_index(model, batch_added, avg_cos, x_query):
	min_score = 999
	min_index = -1
	for i in range(len(x_query)):
		if batch_added[i]:
			continue
		sum = 0
		for feature_vec in x_query[i]:
			sum += abs(model.decision_function(feature_vec))
		avg_dist = sum / len(x_query[i])
		score = _lambda * avg_dist + (1-_lambda) * avg_cos[i]
		if(score < min_score):
			min_score = score
			min_index = i
	return min_index

def get_average_cos(x_query):
	avg_cos = np.zeros(len(x_query))
	for b in range(len(x_query)):
		batch = x_query[b]
		batch_cos = np.zeros(len(batch))
		for i in range(len(batch)):
			for j in range(i+1, len(batch)):
				vector_a = batch[i]
				vector_b = batch[j]
				cos = exp(-_gamma * np.sum((vector_a-vector_b)**2))
				batch_cos[i] = max(batch_cos[i], cos)
				batch_cos[j] = max(batch_cos[j], cos)
		avg_cos[b] = np.average(batch_cos)
	return avg_cos








'''
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
'''

if __name__=='__main__':
	main()