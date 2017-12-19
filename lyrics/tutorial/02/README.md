# 作詞チュートリアル02
## 概要
作詞チュートリアル02では, word embeddingについて学ぶ.

## word embeddingとは
 自然言語をコンピュータにおいて扱うためには, 文書や単語を数値として扱う必要がある. それを実現するための方法を総称してword embeddingという. word embeddingの一例として, word2vecが挙げられる. embeddingとは数学の埋め込みという概念のことである.
 
## 埋め込みとは
 定義を書いてみると...
 > 準同型写像f:A→B(A, B:集合)としたとき, Aとf(A)が同型であれば, fはAをBに埋め込む写像である.  
 
 なにがなんだかわからないので, いろいろな具体例を交えて説明すると,
- 例１（図形）．平面（２次元ユークリッド空間）は立体的空間（３次元ユークリッド空間）に埋め込めます。
- 例２（図形）．球面や球体も、立体的空間に埋め込めます。
- 例３（数）．整数は有理数に埋め込めます。整数や有理数は実数に埋め込めます。整数、有理数、実数は複素数に埋め込めます。
- 例４（集合）．集合Xの部分集合Aは、Xに埋め込まれています。

ということで超おおざっぱに説明すると, 部分集合と見なせることを埋め込むというらしいです.

## word2vecとは
それぞれに単語が対応する場所を1とし, それ以外を0とするようなone-hot表現という方法がある. この方法では単語同士の意味的繋がりを一切無視してしまっている. word2vecは単語を言葉同士の意味的繋がりも考慮した数値(ベクトル)に変換することである.  

例えば,こんなことができる.  
`W(king) - W(man) + W(woman) = W(Queen)`  

なぜこんなことができるかは, 下記のサイトが分り易すぎるので, 参照願いたい.  
[絵で理解するWord2vec](https://qiita.com/Hironsan/items/11b388575a058dc8a46a)

## 実装
### ■概要
このソースコードはkerasが公開している[pretrained_word_emveddings.py](https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py)を参考にしたものである.

### ■とりあえず写経してみよう!
```python

from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model


BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='latin-1')
                t = f.read()
                i = t.find('\n\n')  # skip header
                if 0 < i:
                    t = t[i:]
                texts.append(t)
                f.close()
                labels.append(label_id)

print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
validation_data=(x_val, y_val))
```

## 解説
### ■import
```python
from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preptocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
```
| 名前               | 説明                     |
| :--                | :--                      |
| print_function     | python2.xとpython3.x間で互換性をもたせるもの   |
| os                 | os依存の機能を使うためのモジュール       |
| sys                | インタプリタで使用・管理している変数や, インタプリタの動作に深く関連する関数を定義しているモジュール |
| numpy              | 省略                       |
| Tokenizer          | テキストをベクトル化したり, テキストをシーケンス化したりするクラス |
| pad_sequences      | シーケンスのリストを二次元のリストにする.    |
| to_categorical     | クラスベクトルをcategorical_crossentropyとともに用いるためのバイナルクラス行列に変換する |
| Dense              | 全結合層                     |
| Input              | 入力として受け付けるデータの次元を指定      |
| GlobalMaxPooling1D | 特徴マップ全てに対してMaxPoolingを施す |
| Conv1D             | 畳み込み層                    |
| MaxPooling1D       | 最大値プーリング層                |
| Embedding          | 正の整数を固定次元の密ベクトルに変換する     |
| Model              | 省略                       |

■os.path.join  
賢くパスを繋いでくれる関数.  
(例)
```python
path = os.path.join('src', 'main.py')
print(path)

'''
>>>src/main.py
'''
```

- Tokenizer
  - メソッド
    - fit_on_texts(texts)：学習に使うテキストをいい感じにしてくれる. (内部で何をしているかよくわからなかった)
    - texts_to_sequences(texts)：文章のリストをシーケンスに変換してくれる.

  - 実験
```python
from keras.preprocessing.text import Tokenizer

texts = ['this is a pen', 'he is koki', 'koki is pen']
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
print(sequences)

'''
>>>[[4, 1, 5, 2], [6, 1, 3], [3, 1, 2]]
'''
```
という風にそれぞれの単語が対応する数字が割り振られてる. 見たところ頻出順に数字が割り振られている.  

- pad_sequences
  - 実験
```python
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

texts = ['this is a pen', 'he is koki', 'koki is pen']
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

print(sequences)
print(pad_sequences(sequences))

'''
>>>[[4, 1, 5, 2], [6, 1, 3], [3, 1, 2]]
[[4 1 5 2]
 [0 6 1 3]
 [0 3 1 2]]

'''
```
こんな感じ.

### ■定義
```python
BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'globe.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALUDATION_SPLIT = 0.2
```

|名前|説明|
|:--|:--|
|BASE_DIR|ベースディレクトリを示すパス. ここで指定するディレクトリを基準としてデータセット等を置く. 何も指定しなければカレントディレクトリがベースとなる. |
|GLOVE_DIR|gloveファイルを置くディレクトリを示すパス. gloveとは, スタンフォード大学のプロジェクト名で, Global Vectors for WordRepresentationのことで, word2vecの上位互換だと考えればいい. |
|TEXT_DATA_DIR|テキストデータを置くディレクトリを示すパス. テキストデータはカーネギーメロン大学が提供している20Newsgroupsという約20000文書, 20カテゴリのデータセット.|
|MAX_SEQUENCE_LENGTH|記録するシーケンスの最大数 |
|MAX_NUM_WORDS|記録する単語の最大数|
|EMBEDDING_DIM|gloveファイルの次元数 |
|VALUEDATION_SPLIT|訓練データと教師データを分ける比率|
※gloveファイルとテキストデータはファイルの容量が100MB以上でgithubに保存できなかったので, 下記を参照されたい.
[glove](http://nlp.stanford.edu/projects/glove/)
[text](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html)


### ■embeddingベクトルの準備
```python
print('Indexing word vectors')
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = npasarry(values[1:], dtype='float32)
    embeddings_index[word] = coefs
f.close()
```

まず, gloveファイルを開き, ファイル内の文字列一行一行をlineに代入する. split関数は入力された文字を基準に文字列を分解しリストを返すものだが, 何もしてしなければスペースを基準として分割を行う. gloveの中身がわかりにくいであろうから, 試しにvaluesを出力してみる.

```python
print(values)
'''
>>>['the', '-0.038194', '-0.24487', '0.72812', '-0.39961', '0.083172', '0.043953', '-0.39141', '0.3344', '-0.57545', '0.087459', '0.28787', '-0.06731', '0.30906', '-0.26384', '-0.13231', '-0.20757', '0.33395', '-0.33848', '-0.31743', '-0.48336', '0.1464', '-0.37304', '0.34577', '0.052041', '0.44946', '-0.46971', '0.02628', '-0.54155', '-0.15518', '-0.14107', '-0.039722', '0.28277', '0.14393', '0.23464', '-0.31021', '0.086173', '0.20397', '0.52624', '0.17164', '-0.082378', '-0.71787', '-0.41531', '0.20335', '-0.12763', '0.41367', '0.55187', '0.57908', '-0.33477', '-0.36559', '-0.54857', '-0.062892', '0.26584', '0.30205', '0.99775', '-0.80481', '-3.0243', '0.01254', '-0.36942', '2.2167', '0.72201', '-0.24978', '0.92136', '0.034514', '0.46745', '1.1079', '-0.19358', '-0.074575', '0.23353', '-0.052062', '-0.22044', '0.057162', '-0.15806', '-0.30798', '-0.41625', '0.37972', '0.15006', '-0.53212', '-0.2055', '-1.2526', '0.071624', '0.70565', '0.49744', '-0.42063', '0.26148', '-1.538', '-0.30223', '-0.073438', '-0.28312', '0.37104', '-0.25217', '0.016215', '-0.017099', '-0.38984', '0.87424', '-0.72569', '-0.51058', '-0.52028', '-0.1459', '0.8278', '0.27062']
'''
```
このように`the`という単語とそれに付随するベクトルが入っている. このままでは使いづらいので次の二行で`word`に単語を, `coefs`にベクトルを代入している. さらにもっと使いやすくするために`embeddings_index`ディクショナリに`word`をキーとして, `coefs`をバリューとして呼び出しやすく定義している.


### ■サンプルテキストとラベルの準備
```python
print('Processing text dataset')

texts = []
labels_index = {}
labels = []
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(lebals_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                if sys.version_info < (3, ):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='latin-1')
            
```
listdirとは指定されたディレクトリに存在しているファイルまたはディレクトリ名をリストで返す関数で, 実験してみると.

```python
print(os.listdir(TEXT_DATA_DIR))
>>>['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
```
というように, 20_newsgroupディレクトリ内に存在しているディレクトリを全てリストに保管していることがわかる.  

この一つ一つをnameに代入し, for文でまわしている.  

そのあとは, 例えばひとつ目の`alt.atheism`内にあるファイルつまりは実際のテキストファイルをfpathで指定している. 実際のテキストファイルは数字で名前がつけられているので, `isdigit`関数で数字かどうかを確かめ, ファイルを開いている. 最後に開いたファイルをread関数で読み取り, 変数`t`に渡し, `texts`リストに保管している.

### ■テキストのベクトル化
```python
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape) 
print('Shape o flabel tensor:', labels.shape)
```

上記のプログラムの流れを掴んでもらうために一つ一つ標準出力して中身を確認していく.

```python
print(texts[0])
'''
>>>
Archive-name: atheism/resources
Alt-atheism-archive-name: resources
Last-modified: 11 December 1992
Version: 1.0

                              Atheist Resources

                      Addresses of Atheist Organizations

                                     USA

FREEDOM FROM RELIGION FOUNDATION

Darwin fish bumper stickers and assorted other atheist paraphernalia are
<<<以下略>>>
'''
```
textsには大量のテキストデータが保存されており, 0番目には`alt.atheism`ディレクトリの中の`49960`のファイルの文章が保存されているのがわかる.  


```python
print(sequences[0])
'''
>>>[1237, 273, 1213, 1439, 1071, 1213, 1237, 273, 1439, 192, 2515, 348, 2964, 779, 332, 28, 45, 1628, 1439, 2516, 3, 1628, 2144, 780, 937, 29, 441, 2770, 8854, 4601, 7969, 11979, 5, 12806, 75, 1628, 19, 229, 29, 1, 937, 29, 441, 2770, 6, 1, 118, 558, 2, 90, 106, 482, 3979, 6602, 5375, 1871, 12260, 1632, 17687, 1828, 5101, 1828, 5101, 788, 1, 8854, 4601, 96, 4, 4601, 5455, 64, 1, 751, 563, 1716, 15, 71, 844, 24, 20, 1971, 5, 1, 389, 8854, 744, 1023, 1, 7762, 1300, 2912, 4601, 8, 73, 1698, 6, 1, 118, 558, 2, 1828, 5101, 16500, 13447, 73, 1261, 10982, 170, 66, 6, 1, 869, 2235, 2544, 534, 34, 79, 8854, 4601, 29, 6603, 3388, 264, 1505, 535, 49, 12, 343, 66, 60, 155, 2, 6603, 1043, 1, 427, 8, 73,
<<<以下略>>>
'''
```
sequencesには先程の文章をベクトル化したもので, おそらくそれぞれの文字の対応する数字に変換したもの.  

```python
print(data[0])
'''
>>>[1237, 273, 1213, 1439, 1071, 1213, 1237, 273, 1439, 192, 2515, 348, 2964, 779, 332, 28, 45, 1628, 1439, 2516, 3, 1628, 2144, 780, 937, 29, 441, 2770, 8854, 4601, 7969, 11979, 5, 12806, 75, 1628, 19, 229, 29, 1, 937, 29, 441, 2770, 6, 1, 118, 558, 2, 90, 106, 482, 3979, 6602, 5375, 1871, 12260, 1632, 17687, 1828, 5101, 1828, 5101, 788, 1, 8854, 4601, 96, 4, 4601, 5455, 64, 1, 751, 563, 1716, 15, 71, 844, 24, 20, 1971, 5, 1, 389, 8854, 744, 1023, 1, 7762, 1300, 2912, 4601, 8, 73, 1698, 6, 1, 118, 558, 2, 1828, 5101, 16500, 13447, 73, 1261, 10982, 170, 66, 6, 1, 869, 2235, 2544, 534, 34, 79, 8854, 4601, 29, 6603, 3388, 264, 1505, 535, 49, 12, 343, 66, 60, 155, 2, 6603, 1043, 1, 427, 8, 73,
'''
```
先ほど作成したsequencesをkerasのpad_sequencesメソッドに渡した結果をdataに代入している. pad_sequencesメソッドは各ベクトルの長さが揃っていないときに0を付け足す,もしくはカットすることで同じ長さにそろえるためのメソッドである.


```python
print(labels[0])
'''
>>>[ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
'''
```

見てわかる通りlabelsはどのカテゴリに属しているかを示している.  


### ■訓練データと教師データの準備
```python
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]
```

`indices`にデータ数(20000くらい)までの昇順リストを代入し, そのリストをシャッフルする. そしてシャッフルされた順に`data`と`labels`を並び替える. これで`data`と`labels`の対応関係を崩さないままシャッフルすることができた. 次にtrainデータとvalidationデータに分ける. (validationとは, 「確認」という意味) 今回はデータのうちの0.8をtrainデータにしていることになる.

### ■embedding行列の準備
```python
num_words = min(MAX_NUM_WORDS, len(word_index))
embedding_matrix = np.zeors((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
```

文字に対応する数字をディクショナリ形式で保管している`word_index`からwordを取り出している. その後, 少し前の作ったembedding_indexからwordに対応するベクトルをembedding_vectorに代入し, それをembedding行列の要素に加えていっている. そうしてできたembedding行列をEmbeddingレイヤーに代入している.


### モデル
```python
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequeces)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
              
model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_val, y_val))

```




## 参考
[絵で理解するWord2vec](https://qiita.com/Hironsan/items/11b388575a058dc8a46a)
