import os
import io
import sys
import glob
import random
import pickle

import numpy as np
from utils import wakati, load_data

import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback
from keras.utils.data_utils import get_file



class LyricsNet():

    """
    LyricsNet with LSTM
    """

    def __init__(self, data, args):

        self.maxlen = args.maxlen
        self.step = args.step
        self.lines = args.lines

        self.text = data['text']
        self.chars = data['chars']
        self.char_indices = data['char_indices']
        self.indices_char = data['indices_char']




    def generate(self, model):

        """
        歌詞を生成する関数

        :param model: 学習中もしくは学習済みモデル
        """

        print()
        start_index = random.randint(0, len(self.text) - self.maxlen - 1)
        mecab = MeCab.Tagger("-Ochasen")

        for diversity in [0.4, 0.7, 1.1]:
            print()
            print()
            print('----- diversity:', diversity)

            generated = []
            sentence = self.text[start_index: start_index + self.maxlen]
            generated = sentence

            print('----- Generating with seed: "' + ''.join(sentence) + '"')
            print()

            for i in range(self.lines):

                cnt = 0
                flag = True

                # 一文の生成
                while True:
                    x_pred = np.zeros((1, self.maxlen, len(self.chars)))
                    for t, word in enumerate(sentence):
                        x_pred[0, t, self.char_indices[word]] = 1.

                    # 与えられたsentenceから次の単語を予想
                    preds = model.predict(x_pred, verbose=0)[0]
                    next_index = self.sample(preds, diversity)
                    next_word = self.indices_char[next_index]


                    sentence.append(next_word)
                    sentence.pop(0)

                    if flag and ('名詞' in mecab.parse(next_word).split()[3]):
                        sys.stdout.write(next_word)
                        flag = False
                    elif flag and not(('名詞' in mecab.parse(next_word).split()[3])):
                        continue
                    else:
                        sys.stdout.write(next_word)



                    if (cnt >= 5) and ('助詞' in mecab.parse(next_word).split()[3]):
                        break
                    elif cnt >= 25:
                        break
                    else:
                        cnt += 1

                print()

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


class Logger(keras.callbacks.Callback):

    def __init__(self, model):
        self.model = model

    def on_epoch_end(self):
        LyricsNet.generate()




if __name__ == "__main__":
    import argparse
    import h5py

    # コマンドライン引数を取得
    parser = argparse.ArgumentParser(description="lyricsNet")
    parser.add_argument('--epochs', default=12, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--maxlen', default=8, type=int)
    parser.add_argument('--maxsentence', default=10000, type=int)
    parser.add_argument('--lines', default=20, type=int)
    parser.add_argument('--step', default=2, type=int)
    parser.add_argument('--singer', default='', type=str)
    parser.add_argument('--model_path', default='', type=str)
    args = parser.parse_args()
    print(args)

    # データを生成
    (x, y), data = load_data(args)

    # LyricsNet生成
    LyricsNet = LyricsNet(data, args)

    if os.path.exists(args.model_path):
        model = load_model(args.model_path)

        print()
        print("learned model exists!")

        # 歌詞生成
        LyricsNet.generate(model)
    else:
        # モデルを定義
        model = LyricsNet.createModel(x.shape[1:], y.shape[1])
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=args.lr))

        print("learned model don't exists...\n")

        logger = Logger(model)

        # モデルを実行 (学習と出力)
        model.fit(x, y,
                  batch_size=args.batch_size,
                  epochs=args.epochs,
                  callbacks=[logger])
        LyricsNet.generate(model)
