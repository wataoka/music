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

| 関数名            | 説明                                                                                                                                                                                                          |
| :----             | :---                                                                                                                                                                                                          |
| print_function    | python2系ではprintは関数扱いではなくコマンドのようなものだったので, `print "output"`というふうに括弧なしに書いていた. これを関数扱いするためにprint_functionがある. つまりpypthon3系を使っているものには不必要. |
| Sequential        | 省略                                                                                                                                                                                                          |
| Dense, Activation | 省略                                                                                                                                                                                                          |
| LSTM              | 省略                                                                                                                                                                                                          |
| RMSprop           | 重みを更新するときの関数.[(参考)](https://qiita.com/tokkuman/items/1944c00415d129ca0ee9)                                                                                                                      |
| get_file          | 指定したURLからファイルをダウンロードし, そのファイルへのpathを返す.[(参考)](https://keras.io/ja/utils/data_utils/)                                                                                           |
| numpy             | 省略                                                                                                                                                                                                          |
| random            | 省略                                                                                                                                                                                                          |
| sys               | 省略                                                                                                                                                                                                          |

### ■データセット
```python
path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
```

`https://s3.amazonaws.com/text-datasets/nietzsche.txt`はアマゾンが公開しているニーチェの「善悪の彼岸」(Beyond Good and Evil)という本の全文. これをnietzsche.txtに書き込み, 保存し, 保存した場所を示すパスをpathに代入している. 当然, 保存場所を指定することもできるが, デフォルトでは`~/.keras/datasets/`に保存される. また, 指定する方法は引数に`cache_subdir='path'`を追加する.  

次に保存したnietzsche.txtの中身全部をtextに突っ込む.  

textの中に存在している記号をすべてcharsに代入.  
charsの中身を見てみると...
```python
print(char)
>>>['\n', ' ', '!', '"', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', '[', ']', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ä', 'æ', 'é', 'ë']
```
のようになっている.  

charsに添字を加えたディクショナリを2種類作る.
```python
print(char_indices)
>>>{'\n': 0, ' ': 1, '!': 2, '"': 3, "'": 4, '(': 5, ')': 6, ',': 7, '-': 8, '.': 9, '0': 10, '1': 11, '2': 12, '3': 13, '4': 14, '5': 15, '6': 16, '7': 17, '8': 18, '9': 19, ':': 20, ';': 21, '=': 22, '?': 23, '[': 24, ']': 25, '_': 26, 'a': 27, 'b': 28, 'c': 29, 'd': 30, 'e': 31, 'f': 32, 'g': 33, 'h': 34, 'i': 35, 'j': 36, 'k': 37, 'l': 38, 'm': 39, 'n': 40, 'o': 41, 'p': 42, 'q': 43, 'r': 44, 's': 45, 't': 46, 'u': 47, 'v': 48, 'w': 49, 'x': 50, 'y': 51, 'z': 52, 'ä': 53, 'æ': 54, 'é': 55, 'ë': 56}


print(indices_char)
>>>{0: '\n', 1: ' ', 2: '!', 3: '"', 4: "'", 5: '(', 6: ')', 7: ',', 8: '-', 9: '.', 10: '0', 11: '1', 12: '2', 13: '3', 14: '4', 15: '5', 16: '6', 17: '7', 18: '8', 19: '9', 20: ':', 21: ';', 22: '=', 23: '?', 24: '[', 25: ']', 26: '_', 27: 'a', 28: 'b', 29: 'c', 30: 'd', 31: 'e', 32: 'f', 33: 'g', 34: 'h', 35: 'i', 36: 'j', 37: 'k', 38: 'l', 39: 'm', 40: 'n', 41: 'o', 42: 'p', 43: 'q', 44: 'r', 45: 's', 46: 't', 47: 'u', 48: 'v', 49: 'w', 50: 'x', 51: 'y', 52: 'z', 53: 'ä', 54: 'æ', 55: 'é', 56: 'ë'}
```
