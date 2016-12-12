__author__ = 'mhorowitzgelb'

import os
import matplotlib.pyplot as plt


def main():

	active_step, active_roc = read_in_roc(os.path.expanduser('~/Dropbox/ActiveLearningBackup/eval.txt'))

	random_step, random_roc = read_in_roc(os.path.expanduser('~/Dropbox/ActiveLearningBackup/eval_random.txt'))

	plt.plot(active_step, active_roc , 'k', label= 'Active')
	plt.plot(random_step,random_roc, 'k:', label = 'Random')

	legend = plt.legend(loc = 'bottom right')


	plt.ylabel('ROC Score(AUC)')
	plt.xlabel('Labels Queried')

	plt.show()



def read_in_roc(path):
	f = open(path,'r')

	readHeader = False

	roc = []

	for line in f:
		if not readHeader:
			readHeader = True
			continue
		roc_val = line.split(',')[3]
		roc.append(roc_val)

	step = range(1,len(roc)+1)
	return step, roc

if __name__=='__main__':
	main()