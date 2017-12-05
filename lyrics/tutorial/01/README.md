# 作詞チュートリアル01

## 概要
　作詞チュートリアル01では,kerasが公開しているLSTMを利用した文章生成プログラムを読み解く.

[ソースコード(lstm_text_generator)](https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py)

## モデル
![モデル](https://ai-coordinator.jp/wp-content/uploads/2017/08/LSTM_model.png)

## 解説
### import

'''lstm_text\generation.py
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
improt sys
'''

