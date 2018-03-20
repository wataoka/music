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
import numpy as np
import random
import sys
import io



class LyricsNet():

    """
    LyricsNet with LSTM
    """

    def __init__(self, text, chars, char_indices, indices_char):

        self.text = text
        self.chars = chars
        self.char_indices = char_indices
        self.indices_char = indices_char

        self.maxlen = 40
        self.step = 3


    def createModel(self, input_shape, n_class):

        """
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

        # 各エポック終了後に呼び出され, 生成したテキストを標準出力する.

        print()
        print('----- Generating text after Epoch: %d' % epoch)

        start_index = random.randint(0, len(text) - self.maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('----- diversity:', diversity)

            generated = ''
            sentence = self.text[start_index: start_index + self.maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(400):
                x_pred = np.zeros((1, self.maxlen, len(self.chars)))
                for t, self.char in enumerate(sentence):
                    x_pred[0, t, self.char_indices[self.char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, diversity)
                next_char = self.indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()


    def sample(self, preds, temperature=1.0):

        # 単語リストを正規化し, 確率的に単語を選択する.

        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)



def load_data():

    path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
    with io.open(path, encoding='utf-8') as f:
           text = f.read().lower()

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
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    return (x, y), (text, chars, char_indices, indices_char)



if __name__ == "__main__":

    # データの作成
    (x, y), (text, chars, char_indices, indices_char) = load_data()

    # LyricsNet生成
    LyricsNet = LyricsNet(text, chars, char_indices, indices_char)

    # モデルを定義
    model = LyricsNet.createModel(x.shape[1:], y.shape[1])
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))

    # モデルを実行 (学習と出力)
    model.fit(x, y,
              batch_size=128,
              epochs=60,
              callbacks=[LambdaCallback(on_epoch_end=LyricsNet.on_epoch_end)])
