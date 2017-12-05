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
hitherto spoken with respect, what questions has this will to truth not
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
for i in range(0, len(text) maxlen, step):
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
