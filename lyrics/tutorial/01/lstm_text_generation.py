from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import numpy as np
import random
import sys

path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])




print("---sentences0---")
print(sentences[0])
print("---next_chars0---")
print(next_chars[0])
print("---sentences1---")
print(sentences[1])
print("---next_chars1---")
print(next_chars[1])
print("---sentences2---")
print(sentences[2])
print("---next_chars2---")
print(next_chars[2])
print("---sentences3---")
print(sentences[3])
print("---next_chars3---")
print(next_chars[3])

