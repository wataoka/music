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

`https://s3.amazonaws.com/text-datasets/nietzsche.txt`はアマゾンが公開しているニーチェの「善悪の彼岸」(Beyond Good and Evil)という本の全文. これをnietzsche.txtに書き込み, 保存し, 保存した場所を示すパスをpathに代入している. 当然, 保存場所を指定することもできるが, デフォルトでは`~/.keras/datasets/`に保存されることになっている. また, 指定する方法は引数に`cache_subdir='path'`を追加する.  

次に保存したnietzsche.txtの中身をtextに突っ込む.  
textの中をちょっとだけ見てみると...

```python
print(text[0: 10000])

'''
>>>preface


supposing that truth is a woman--what then? is there not ground
for suspecting that all philosophers, in so far as they have been
dogmatists, have failed to understand women--that the terrible
seriousness and clumsy importunity with which they have usually paid
their addresses to truth, have been unskilled and unseemly methods for
winning a woman? certainly she has never allowed herself to be won; and
at present every kind of dogma stands with sad and discouraged mien--if,
indeed, it stands at all! for there are scoffers who maintain that it
has fallen, that all dogma lies on the ground--nay more, that it is at
its last gasp. but to speak seriously, there are good grounds for hoping
that all dogmatizing in philosophy, whatever solemn, whatever conclusive
and decided airs it has assumed, may have been only a noble puerilism
and tyronism; and probably the time is at hand when it will be once
and again understood what has actually sufficed for the basis of such
imposing and absolute philosophical edifices as the dogmatists have
hitherto reared: perhaps some popular superstition of immemorial time
(such as the soul-superstition, which, in the form of subject- and
ego-superstition, has not yet ceased doing mischief): perhaps some
play upon words, a deception on the part of grammar, or an
audacious generalization of very restricted, very personal, very
human--all-too-human facts. the philosophy of the dogmatists, it is to
be hoped, was only a promise for thousands of years afterwards, as was
astrology in still earlier times, in the service of which probably more
labour, gold, acuteness, and patience have been spent than on any
actual science hitherto: we owe to it, and to its "super-terrestrial"
pretensions in asia and egypt, the grand style of architecture. it seems
that in order to inscribe themselves upon the heart of humanity with
everlasting claims, all great things have first to wander about the
earth as enormous and awe-inspiring caricatures: dogmatic philosophy has
been a caricature of this kind--for instance, the vedanta doctrine in
asia, and platonism in europe. let us not be ungrateful to it, although
it must certainly be confessed that the worst, the most tiresome,
and the most dangerous of errors hitherto has been a dogmatist
error--namely, plato's invention of pure spirit and the good in itself.
but now when it has been surmounted, when europe, rid of this nightmare,
can again draw breath freely and at least enjoy a healthier--sleep,
we, whose duty is wakefulness itself, are the heirs of all the strength
which the struggle against this error has fostered. it amounted to
the very inversion of truth, and the denial of the perspective--the
fundamental condition--of life, to speak of spirit and the good as plato
spoke of them; indeed one might ask, as a physician: "how did such a
malady attack that finest product of antiquity, plato? had the wicked
socrates really corrupted him? was socrates after all a corrupter of
youths, and deserved his hemlock?" but the struggle against plato,
or--to speak plainer, and for the "people"--the struggle against
the ecclesiastical oppression of millenniums of christianity (for
christianity is platonism for the "people"), produced in europe
a magnificent tension of soul, such as had not existed anywhere
previously; with such a tensely strained bow one can now aim at the
furthest goals. as a matter of fact, the european feels this tension as
a state of distress, and twice attempts have been made in grand style to
unbend the bow: once by means of jesuitism, and the second time by means
of democratic enlightenment--which, with the aid of liberty of the press
and newspaper-reading, might, in fact, bring it about that the spirit
would not so easily find itself in "distress"! (the germans invented
gunpowder--all credit to them! but they again made things square--they
invented printing.) but we, who are neither jesuits, nor democrats,
nor even sufficiently germans, we good europeans, and free, very free
spirits--we have it still, all the distress of spirit and all the
tension of its bow! and perhaps also the arrow, the duty, and, who
knows? the goal to aim at....

sils maria upper engadine, june, 1885.




chapter i. prejudices of philosophers


1. the will to truth, which is to tempt us to many a hazardous
enterprise, the famous truthfulness of which all philosophers have
laid before us! what strange, perplexing, questionable questions! it is
already a long story; yet it seems as if it were hardly commenced. is
it any wonder if we at last grow distrustful, lose patience, and turn
impatiently away? that this sphinx teaches us at last to ask questions
ourselves? who is it really that puts questions to us here? what really
is this "will to truth" in us? in fact we made a long halt at the
question as to the origin of this will--until at last we came to an
absolute standstill before a yet more fundamental question. we inquired
about the value of this will. granted that we want the truth: why not
rather untruth? and uncertainty? even ignorance? the problem of the
value of truth presented itself before us--or was it we who presented
ourselves before the problem? which of us is the oedipus here? which
the sphinx? it would seem to be a rendezvous of questions and notes of
interrogation. and could it be believed that it at last seems to us as
if the problem had never been propounded before, as if we were the first
to discern it, get a sight of it, and risk raising it? for there is risk
in raising it, perhaps there is no greater risk.

2. "how could anything originate out of its opposite? for example, truth
out of error? or the will to truth out of the will to deception? or the
generous deed out of selfishness? or the pure sun-bright vision of the
wise man out of covetousness? such genesis is impossible; whoever dreams
of it is a fool, nay, worse than a fool; things of the highest
value must have a different origin, an origin of their own--in this
transitory, seductive, illusory, paltry world, in this turmoil of
delusion and cupidity, they cannot have their source. but rather in
the lap of being, in the intransitory, in the concealed god, in the
'thing-in-itself--there must be their source, and nowhere else!"--this
mode of reasoning discloses the typical prejudice by which
metaphysicians of all times can be recognized, this mode of valuation
is at the back of all their logical procedure; through this "belief" of
theirs, they exert themselves for their "knowledge," for something that
is in the end solemnly christened "the truth." the fundamental belief of
metaphysicians is the belief in antitheses of values. it never occurred
even to the wariest of them to doubt here on the very threshold (where
doubt, however, was most necessary); though they had made a solemn
vow, "de omnibus dubitandum." for it may be doubted, firstly, whether
antitheses exist at all; and secondly, whether the popular valuations
and antitheses of value upon which metaphysicians have set their
seal, are not perhaps merely superficial estimates, merely provisional
perspectives, besides being probably made from some corner, perhaps from
below--"frog perspectives," as it were, to borrow an expression current
among painters. in spite of all the value which may belong to the true,
the positive, and the unselfish, it might be possible that a higher
and more fundamental value for life generally should be assigned to
pretence, to the will to delusion, to selfishness, and cupidity. it
might even be possible that what constitutes the value of those good and
respected things, consists precisely in their being insidiously
related, knotted, and crocheted to these evil and apparently opposed
things--perhaps even in being essentially identical with them. perhaps!
but who wishes to concern himself with such dangerous "perhapses"!
for that investigation one must await the advent of a new order of
philosophers, such as will have other tastes and inclinations, the
reverse of those hitherto prevalent--philosophers of the dangerous
"perhaps" in every sense of the term. and to speak in all seriousness, i
see such new philosophers beginning to appear.

3. having kept a sharp eye on philosophers, and having read between
their lines long enough, i now say to myself that the greater part of
conscious thinking must be counted among the instinctive functions, and
it is so even in the case of philosophical thinking; one has here to
learn anew, as one learned anew about heredity and "innateness." as
little as the act of birth comes into consideration in the whole process
and procedure of heredity, just as little is "being-conscious" opposed
to the instinctive in any decisive sense; the greater part of the
conscious thinking of a philosopher is secretly influenced by his
instincts, and forced into definite channels. and behind all logic and
its seeming sovereignty of movement, there are valuations, or to speak
more plainly, physiological demands, for the maintenance of a definite
mode of life for example, that the certain is worth more than the
uncertain, that illusion is less valuable than "truth" such valuations,
in spite of their regulative importance for us, might notwithstanding be
only superficial valuations, special kinds of _niaiserie_, such as may
be necessary for the maintenance of beings such as ourselves. supposing,
in effect, that man is not just the "measure of things."

4. the falseness of an opinion is not for us any objection to it: it is
here, perhaps, that our new language sounds most strangely. the
question is, how far an opinion is life-furthering, life-preserving,
species-preserving, perhaps species-rearing, and we are fundamentally
inclined to maintain that the falsest opinions (to which the synthetic
judgments a priori belong), are the most indispensable to us, that
without a recognition of logical fictions, without a comparison of
reality with the
'''
```
という風に, ニーチェの文章がちゃんと出てきた.  

textの中に存在している記号をすべてcharsに代入.  
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

### ■前処理
```python
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))
```

まず1字目〜40字目をsentencesに追加し, 次に4字目〜44字目までをsentencesに追加する. このように3字飛ばしづつの文章をsentenceに加え, そのつぎに来る文字をnext_charsに追加する.  

実際に確かめてみると...

```python
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

'''
>>>
---sentences0---
preface


supposing that truth is a woma
---next_chars0---
n
---sentences1---
face


supposing that truth is a woman--
---next_chars1---
w
---sentences2---
e


supposing that truth is a woman--wha
---next_chars2---
t
---sentences3---

supposing that truth is a woman--what t
---next_chars3---
h
'''

```
となる. 全文と見比べてみればわかるとおり, 文章とその次にくる文字がそれぞれsentencesとnext_charsにきちんと入っていることがわかる.

### ■教師データの準備

#### 0を敷き詰める
```python
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
```
先ほど作った大量の文章をone-hotエンコーディングするために, まずは0を敷き詰める.(正確にはFalseを敷き詰めいてる)  

大きさがデータの形がわかりにくかったのでnumpyに調べてもらった.
```python
print(np.shape(x))
print(np.shape(y))
'''
>>>(200285, 40, 57)
(200285, 57)
'''
```
200285が大量に作った文章の数  
57が記号の数で, あとで一文字一文字対応する場所を1に変更する.  
そして, xに関しては40個の文字列で一セットなので(200284, 40, 57)となっている.  


#### 対応する場所を1にする
```python
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
```
ここでさきほど作った`char_indices`ディクショナリが大活躍. このディクショナリのkeyに文字を与えるとそれに対応する場所を示すインデックスを返してくれる.これを利用して, 先ほどの57のところに`char_indices[char]`と`char_indices[next_char[i]]`をぶち込んでやれば全てがうまく行く.  

一応`char_indices`の機能を確認しておく.
```python
print(chars)
print(char_indices['!'])
'''
>>>['\n', ' ', '!', '"', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', '[', ']', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ä', 'æ', 'é', 'ë']
2
'''
```
きちんと`!`に対応する場所を示すインデックスである`2`を返してくれている.  


### ■モデルの設計
```python
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
```
こんな感じ.
![モデル](https://ai-coordinator.jp/wp-content/uploads/2017/08/LSTM_model.png)  



### ■sample関数の定義
```python
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
```
入力された確率のリストを正規化し, 確率的に文字を選んでくる関数.  
predsリストの中には次に来る文字の確率が格納されている. その確率の最大値を選んで来るという決定的な方法ではなく, 確率に従って確率的に選んで来る. ちなみに前者を貪欲法と言うらしい.  
実際に動かしてみると,

```python
preds = [  6.05958747e-04   1.46672153e-03   7.70759361e-05   3.32636992e-03
   4.41987904e-05   6.14247168e-04   3.37334495e-05   4.34828107e-04
   5.08925878e-04   2.06383556e-04   9.33959655e-07   4.21795667e-05
   2.38951998e-05   7.14040107e-06   2.37641734e-06   2.36847359e-06
   7.78792401e-06   1.32757123e-06   2.81304779e-06   1.19497363e-05
   1.57066213e-04   1.23525417e-04   8.68624265e-05   7.96350214e-05
   3.08299741e-06   7.77315745e-06   1.17005166e-05   1.78474337e-01
   2.85854023e-02   2.75644697e-02   1.16788959e-02   1.75518319e-02
   3.10252998e-02   6.06567273e-03   1.79951563e-02   6.03913851e-02
   2.44643539e-04   1.43562758e-03   1.72499064e-02   3.17009166e-02
   1.37064010e-02   1.83961838e-01   3.21779065e-02   3.37578967e-04
   8.14340636e-03   3.24097723e-02   2.00572893e-01   2.68001724e-02
   4.29536868e-03   5.60857765e-02   7.83014184e-06   3.58585967e-03
   6.08548798e-05   2.36636133e-09   2.13360085e-09   2.09655981e-09
   2.89875124e-09]

ans = sample(preds)
print(ans)

'''
>>>27
'''
```
リスト内の27番目にある`1.78474337e-01`が選ばれた. 最大値は多分36番目の`2.00572893e-01 `なので, 単に最大値を選択しているのではなく確率的に選択してきているのがわかる.



### ■学習


```python
for iteration in renge(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x, y,
              batch_size=128,
              epochs=1
    )
    
    start_index = random.randint(0, len(text) - maxlen - 1)
    
    for diversity in [0.2, 0.5, 1.0, 1.2]:
       print()
       print('----- diversity:', diversity)
       generated = ''
       sentence = text[start_index: start_index + maxlen]
       generated += sentence
       print('----- Generating with seed: "' + sentence + '"')
       sys.stdout.write(generated)
       
       for i in range(400):
           x_pred = np.zeros((1, maxlen, len(chars)))
           for t, char in enumerate(sentence):
               x_pred[0, t, char_indices[char]] = 1
           
           preds = model.predict(x_pred, verbose=0)[0]
           next_index = sample(preds, diversity)
           next_char = indices_char[next_index]
           
           generated += next_char
           sentence = sentence[1:] + next_char
           
           sys.stdout.write(next_char)
           sys.stdout.flush()
       print()
```

#### 訓練
訓練は`model.fit`の一行で行われている.  
for文で定義されている通り, 60回学習する.  

#### テスト
訓練が1度終われば, すぐにテストを行っている.  
テストの方法は,  
1. 適当な40字の文章を本文から取ってくる.
2. その40字を`sentence`代入する.
3. `sentence`から予想される次の1文字を`next_char`に代入する.
4. `next_char`を`sentence`に加える.
5. 逐一, 出来栄えを出力する.
6. これを400字作るまでくりかえす.

## 結果
60回学習するのはめんどくさかったので6回学習させた結果.

### 1回目
はじめの40字
>repulsive superstition--he dealt with
a
生成した続きの400字
>nd the sensental sension of the still and the sention of the still and the more the still and the still the still the stringer which the still the stand of the man from the men the sense of the will the sensent in the still the still man the sense of the the subligent the still the still the still the sense of the still the sense of the more the more and the sensental the still the still the still

### 2回目
はじめの40字
>e would it give that
>it would not contin

生成した続きの400字
>unal proposition of the more of the concertal strength of the stand it is the most strive of the understand the strange of the expression of every man estime and in the strength and the man in the fact of the strength of the strange of the master of the more of the master of the strange and the more concerning and and and only the most stand of the strange of the strength of the staniness of the p

### 3回目
はじめの40字
>ork and invention of
>france; the europea

生成した続きの400字
>n in the most in the conscience of the state and in the respection of a sufficient of the sense of the self in the substation of the subjection of the self interpariest of the same the same a sufficient of all philosophy in the same the sense of the most conscience of a probably the same the same the same a sufficient of the sense of the same the consciously and in the strength in the most conscie

### 4回目
はじめの40字
>pe of new
>germanism is covetous of quite

生成した続きの400字
>the sense of the most and soul the most and and it is a man is the sentiment of the possible and the most reason and the sense it is the most and and of the sense and the most and the sense of the sense of the sublimation of the serve and the sense of the sertion and the most and the world of the most and intellectual and and as the sert of the sense of the sense of the strength of the servess an

### 5回目
はじめの40字
>ems to me that in this case schopenhauer

生成した続きの400字
>to the state of the strength of the strends of the stronger of the expressed to the stronger in the strended to the sinceration of the same as the sincess of the problem of the strongess of the state of the strongess of the stronger and all the soul of the strongess and problem of the stronger to the sense of the strongess and the stronger of the strongess and the stronger and the same shard of t

### 6回目
はじめの40字
>ment (a kind of rococo of taste in every

生成した続きの400字
> strong the probably the morality and a surpose and the concealis and a propers its spirits of the powerful to the self-desire and a proper the subtlety and perhaps the state of the soul of the struggle of the concealing the most conscience of the soul of the decises that the procession of the spirits, and a something and such a still a proper the more and a still as the spirits and a concealistic


## 考察
文章をしっかり書けているわけではないが単語を覚えられたのは見て取れる.
