import numpy as np
import pickle
import sys
import random


def get_chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]





f = open('./data/all_data_0330_164.svm', 'r')


y = []
x = []

for line in f:
    split = line.split()
    y.append(int(split[0]))
    features = []
    for str in split[1:]:
        features.append(float(str.split(':')[1]))
    x.append(features)



x = np.array(x,dtype=np.float32)
y = np.array(y, dtype=np.int)



chunks_x = np.array_split(x,263)
chunks_y = np.array_split(y,263)


index_shuf = list(range(263))

random.shuffle(index_shuf)

shuffle_chunks_x, shuffle_chunks_y = [], []

for i in index_shuf:
    shuffle_chunks_x.append(chunks_x[i])
    shuffle_chunks_y.append(chunks_y[i])


train_x = shuffle_chunks_x[40:]
train_y = shuffle_chunks_y[40:]

test_x = shuffle_chunks_x[:40]
test_y = shuffle_chunks_y[:40]


print(len(train_x))
print(len(test_x))





pickle.dump((train_x,train_y), open('data/train.pkl','wb'))

pickle.dump((test_x,test_y), open('data/test.pkl','wb'))




