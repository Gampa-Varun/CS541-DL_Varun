import bcolz
import numpy as np
import pickle
words = []
idx = 0
word2idx = {}
glove_path = 'gloveData'
vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.100.dat', mode='w')

with open(f'{glove_path}/glove.6B.100d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float64)
        vectors.append(vect)
    
vectors = bcolz.carray(vectors[1:].reshape((400000, 100)), rootdir=f'{glove_path}/6B.100.dat', mode='w')
vectors.flush()
pickle.dump(words, open(f'{glove_path}/6B.100_words.pkl', 'wb'))
pickle.dump(word2idx, open(f'{glove_path}/6B.100_idx.pkl', 'wb'))