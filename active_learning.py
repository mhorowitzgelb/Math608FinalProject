import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import os
from math import exp


dropbox_dir = os.path.expanduser("~/Dropbox/ActiveLearningBackup/")

_lambda = 0.5
_gamma = 0.0008

random_query = True

eval_path = dropbox_dir+'eval' + ( '_random' if random_query else '') +'.txt'
query_path = dropbox_dir+'query'+('_random' if random_query else '')+'.txt'

def main():
	print(dropbox_dir)
	x_train , y_train = pickle.load(open('data/init.pkl','rb'))

	x_query, y_query = pickle.load(open('data/query.pkl','rb'))

	x_test, y_test = pickle.load(open('data/test.pkl','rb'))


	x_train = np.concatenate(x_train)
	y_train = np.concatenate(y_train)

	x_test = np.concatenate(x_test)
	y_test = np.concatenate(y_test)



	active_learning_step = 0

	model = SVC(kernel='rbf', gamma=_gamma)

	model.fit(x_train,y_train)
	evaluate_model(model,active_learning_step,x_test, y_test)


	batch_cos = get_batch_cos(x_query)

	batch_added = np.zeros(len(x_query))

	with open(eval_path,'w') as eval_f:
		eval_f.write('precision,recall,f1,roc\n')

	with open(query_path,'w') as query_f:
		query_f.write('query_index\n')



	while active_learning_step <= len(x_query):
		active_learning_step += 1
		print("Running active learning step ", active_learning_step, 'of', len(x_query))
		print("Selecting next batch")
		query_index = get_random_query_index(batch_added) if random_query else get_query_index(model,batch_added,batch_cos,x_query)
		print("Selected query batch:", query_index)
		with open(query_path,'a') as query_f:
			query_f.write('{}\n'.format(query_index))

		x_train = np.concatenate([x_train,x_query[query_index]])
		y_train = np.concatenate([y_train,y_query[query_index]])
		batch_added[query_index] = True
		model = model.fit(x_train,y_train)
		evaluate_model(model,active_learning_step,x_test,y_test)








def evaluate_model(model,learning_step, x_test, y_test):
	pickle.dump(model, open(dropbox_dir+'model_{}'.format(learning_step) + ('_random.pkl' if random_query else '.pkl'),'wb'))
	pred = model.predict(x_test)
	print(pred)

	print("Evaluating model at learning step", learning_step)
	precision = precision_score(y_true=y_test, y_pred=pred)
	recall = recall_score(y_true=y_test,y_pred=pred)
	f1 = f1_score(y_test,pred)



	scores = model.decision_function(x_test)

	roc = roc_auc_score(y_test,scores)

	print("precision:", precision, "recall:" , recall , "f1:" , f1, "roc" , roc)

	with open(eval_path, 'a') as eval_f:
		eval_f.write('{},{},{},{}\n'.format(precision,recall,f1,roc))

def get_query_index(model, batch_added, batch_cos, x_query):
	max_score = -999
	max_index = -1
	for i in range(len(x_query)):
		batch = x_query[i]
		if batch_added[i]:
			continue

		dist = np.abs(model.decision_function(batch))
		mask = dist < 1

		dist_score = np.sum(1 - dist[mask])
		cos_score = np.sum(1- batch_cos[i][mask])


		score = _lambda * dist_score + (1-_lambda) * cos_score
		if(score > max_score):
			max_score = score
			max_index = i
	return max_index

def get_random_query_index(batchAdded):
	mask = batchAdded == 0
	indices = np.array(range(len(batchAdded)))[mask]
	return np.random.choice(indices)



def get_batch_cos(x_query):
	batch_cosines = []
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
		batch_cosines.append(batch_cos)
	return batch_cosines








if __name__=='__main__':
	main()
