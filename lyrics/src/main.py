import os
import io
import glob
import h5py
import pickle
import argparse

from Logger import Logger
from utils import wakati, load_data

import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback
from keras.utils.data_utils import get_file



def createModel(input_shape, n_class):

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



if __name__ == "__main__":

    # コマンドライン引数を取得
    parser = argparse.ArgumentParser(description="lyricsNet")

    parser.add_argument('--epochs', default=12, type=int)
    parser.add_argument('--batch_size', default=100, type=int)

    parser.add_argument('--lr', default=0.01, type=float)

    parser.add_argument('--maxlen', default=8, type=int)
    parser.add_argument('--maxsentence', default=10000, type=int)
    parser.add_argument('--lines', default=20, type=int)
    parser.add_argument('--step', default=2, type=int)

    parser.add_argument('--singer', default='AI', type=str)
    parser.add_argument('--model_path', default='', type=str)
    args = parser.parse_args()

    # データを生成
    (x, y), data = load_data(args)

    if os.path.exists(args.model_path):
        print("\n学習済みモデルが存在します!")
        model = load_model(args.model_path)

    else:
        # モデルを定義
        model = createModel(x.shape[1:], y.shape[1])
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=args.lr))

        logger = Logger(model, args, data)

        print("学習を開始します。\n")
        model.fit(x, y,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  callbacks=[logger])
