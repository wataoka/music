# 作詞チュートリアル01

## 概要
　作詞チュートリアル01では,kerasが公開しているLSTMを利用した文章生成プログラムを読み解く.

[ソースコード(lstm_text_generator)](https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py)

## モデル
![モデル](https://ai-coordinator.jp/wp-content/uploads/2017/08/LSTM_model.png)

## 解説
### ■import

```python
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
```

・print_function
python2系ではprintは関数扱いではなくコマンドのようなものだったので, `print output`というふうに括弧なしに書いていた. これを関数扱いするためにprint_functionがある. つまりpypthon3系を使っているものには不必要.

・Sequential
省略

・Dense, Activation
省略

・LSTM
省略

・RMSprop
重みを更新するときの関数.
[参考](https://qiita.com/tokkuman/items/1944c00415d129ca0ee9)

・get_file
指定したURLからファイルをダウンロードし, そのファイルへのpathを返す.
[参考](https://keras.io/ja/utils/data_utils/)

・numpy
省略

・random
省略

・sys
省略
