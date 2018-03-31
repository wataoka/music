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
import glob
import numpy as np
import random
import sys
import io
import os
from tqdm import tqdm
import pickle

import lyricsNet
from lyricsNet import wakati, load_data


def load_data_singer(args):

    """
    教師データをロードする関数.

    :param args: maxlenとstepなどを格納したargparseオブジェクト
    :return (x, y): 教師データ
    :return data: 文章生成に必要なデータ
    """

    text = []

    maxlen = args.maxlen
    step = args.step
    maxsentence = args.maxsentence
    singer = args.singer

    import json
    file_list = glob.glob('./data/json/*.json')
    for file in file_list:
        f = open(file, 'r')
        j = json.load(f)
        f.close
        for song in j['data']:
            if song['singer'] == singer:
                wordlist = wakati(song['lyrics'].replace('\n', ''))
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
    import h5py

    # コマンドライン引数を取得
    parser = argparse.ArgumentParser(description="lyricsNet")
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--length', default=100, type=int)
    parser.add_argument('--maxlen', default=5, type=int)
    parser.add_argument('--maxsentence', default=10000, type=int)
    parser.add_argument('--step', default=1, type=int)
    parser.add_argument('--singer', default='福山雅治', type=str)
    args = parser.parse_args()
    print(args)

    # データを生成
    (x, y), data = load_data_singer(args)

    # LyricsNet生成
    LyricsNet = LyricsNet(data, args)

    if os.path.exists('./model/model_'+args.singer+'.h5'):
        from keras.models import load_model
        model = load_model('./model/model_' + args.singer + '.h5')

        print()
        print("learned model exists!")

        # 歌詞生成
        LyricsNet.generate(model)
    else:
        # モデルを定義
        model = LyricsNet.createModel(x.shape[1:], y.shape[1])
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=args.lr))

        # モデルを実行 (学習と出力)
        model.fit(x, y,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    callbacks=[LambdaCallback(on_epoch_end=LyricsNet.on_epoch_end)])
        model.save('./model/model_' + args.singer + '.h5')
