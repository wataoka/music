'''
LSTMを用いて歌詞を自動生成する.
'''

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import MeCab
import glob
import numpy as np
import random
import sys
import io
import os
from tqdm import tqdm
import pickle



class LyricsNet():

    """
    LyricsNet with LSTM
    """

    def __init__(self, data, args):

        self.maxlen = args.maxlen
        self.step = args.step
        self.text = data['text']
        self.chars = data['chars']
        self.char_indices = data['char_indices']
        self.indices_char = data['indices_char']



    def createModel(self, input_shape, n_class):

        """
        モデルを生成する関数.

        :param input_shape: データの形 ([文章の数, 文章の長さ, 文字の数])
        :param n_class: クラス数 (文字の数)
        :return: LyricsNetモデル (Single LSTM)
        """

        model = Sequential()
        model.add(LSTM(128, input_shape=input_shape))
        model.add(Dense(n_class))
        model.add(Activation('softmax'))

        return model


    def on_epoch_end(self, epoch, logs):

        """
        各エポック終了後に呼び出され, 生成したテキストを標準出力する関数.

        :param epoch: 現在のエポック数
        :param logs: ログ
        """

        print()
        print('----- Generating text after Epoch: %d' % epoch)

        start_index = random.randint(0, len(self.text) - self.maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('----- diversity:', diversity)

            generated = []
            sentence = self.text[start_index: start_index + self.maxlen]
            generated = sentence

            print('----- Generating with seed: "' + ''.join(sentence) + '"')
            sys.stdout.write(''.join(generated))

            for i in range(400):
                x_pred = np.zeros((1, self.maxlen, len(self.chars)))
                for t, word in enumerate(sentence):
                    x_pred[0, t, self.char_indices[word]] = 1.

                # 与えられたsentenceから次の単語を予想
                preds = model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, diversity)
                next_word = self.indices_char[next_index]

                sentence.append(next_word)
                sentence.pop(0)

                sys.stdout.write(next_word)
                sys.stdout.flush()
            print()


    def sample(self, preds, temperature=1.0):

        """
        単語リストを正規化し, 確率的に単語を選択する関数.

        :param preds: 確率が格納されたリスト
        :param temperature: 温度と呼ばれる学習パラメータで, 高ければ選択の確率が平滑化される.
        :return: 選択した単語のindex
        """

        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)




def wakati(text):

    """
    分かち書きをする関数.

    :param text: 分かち書きをしたい文章
    :return: 分かち書きされた単語のリスト
    """

    t = MeCab.Tagger("-Owakati")
    m = t.parse(text)
    result = m.rstrip(" \n").split(" ")
    return result


def load_data(args):

    """
    教師データをロードする関数.

    :param args: maxlenとstepなどを格納したargparseオブジェクト
    :return (x, y): 教師データ
    :return data: 文章生成に必要なデータ
    """

    maxlen = args.maxlen
    step = args.step
    maxsentence = args.maxsentence

    file_list = glob.glob('./data/txt/*.txt')
    text = []
    print("wakati sentense.")
    for i, file in enumerate(tqdm(file_list)):
        if i%70 == 0:
            src = open(file, 'r').read()
            wordlist = wakati(src)
            for word in wordlist:
                text.append(word)


    print("sorting text.")
    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(tqdm(chars)))
    indices_char = dict((i, c) for i, c in enumerate(tqdm(chars)))

    sentences = []
    next_chars = []
    for i in tqdm(range(0, len(text) - maxlen, step)):
        if len(sentences) >= maxsentence:
            break
        else:
            sentences.append(text[i: i + maxlen])
            next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(tqdm(sentences)):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1


    data = {}
    data['text'] = text
    data['chars'] = chars
    data['char_indices'] = char_indices
    data['indices_char'] = indices_char

    return (x, y), data



if __name__ == "__main__":
    import argparse

    # コマンドライン引数を取得
    parser = argparse.ArgumentParser(description="lyricsNet")
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--maxlen', default=10, type=int)
    parser.add_argument('--maxsentence', default=10000, type=int)
    parser.add_argument('--step', default=2, type=int)
    args = parser.parse_args()
    print(args)

    # データを生成
    (x, y), data = load_data(args)

    # LyricsNet生成
    LyricsNet = LyricsNet(data, args)

    # モデルを定義
    model = LyricsNet.createModel(x.shape[1:], y.shape[1])
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=args.lr))

    # モデルを実行 (学習と出力)
    model.fit(x, y,
              batch_size=args.batch_size,
              epochs=args.epochs,
              callbacks=[LambdaCallback(on_epoch_end=LyricsNet.on_epoch_end)])
