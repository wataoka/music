import sys
import MeCab
import random
import numpy as np

import keras

class Logger(keras.callbacks.Callback):

    def __init__(self, model, args, data):
        self.model = model

        self.maxlen = args.maxlen
        self.step = args.step
        self.lines = args.lines

        self.text = data['text']
        self.chars = data['chars']
        self.char_indices = data['char_indices']
        self.indices_char = data['indices_char']

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

    def on_epoch_end(self, epoch, logs={}):

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
                    preds = self.model.predict(x_pred, verbose=0)[0]
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
