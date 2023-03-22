from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import sys

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchinfo import summary
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from torchtext.data import get_tokenizer
import bcolz
import pickle
import numpy as np

tokenizer = get_tokenizer("basic_english") ## We'll use tokenizer available from PyTorch



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


glove_path = 'gloveData'

vectors = bcolz.open(f'{glove_path}/6B.100.dat')[:]
words = pickle.load(open(f'{glove_path}/6B.100_words.pkl', 'rb'))
word2idx = pickle.load(open(f'{glove_path}/6B.100_idx.pkl', 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}





# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s



def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        pairs = [list(p) for p in pairs]
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs



MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[0].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]



def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words, '\n')



    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'fra', reverse = False)


matrix_len = len(input_lang.index2word)
weights_matrix = np.zeros((matrix_len, 100))
words_found = 0
emb_dim = 100
i = 0

for key in (input_lang.index2word):
    try:  
        weights_matrix[i] = glove[input_lang.index2word[key]]
        words_found += 1
    except KeyError:
        if input_lang.index2word[key] == 'SOS':
            weights_matrix[i] = np.zeros((emb_dim, ))
            weights_matrix[i][0] = 1
        elif input_lang.index2word[key] == 'EOS':
            weights_matrix[i] = np.zeros((emb_dim, ))
            weights_matrix[i][0] = emb_dim

        else:
            print(f'Word not found:{input_lang.index2word[key]}')
            weights_matrix[i] = np.zeros((emb_dim, ))
    i = i+1

print('\n')


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = np.shape(weights_matrix)
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': torch.from_numpy(weights_matrix)})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

       
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
