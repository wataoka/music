import glob
import MeCab

import numpy as np
from tqdm import tqdm

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
    singer = args.singer

    import json
    file_list = glob.glob('./data/json/*.json')
    text = []
    for file in file_list:
        f = open(file, 'r')
        j = json.load(f)
        f.close()
        for song in j['data']:
            if singer == '':
                wordlist = wakati(song['lyrics'].replace('\n', ''))
                for word in wordlist:
                    text.append(word)
            elif singer == song['singer']:
                wordlist = wakati(song['lyrics'].replace('\n', ''))
                for word in wordlist:
                    text.append(word)

    if text == '':
        print(singer, 'さんの曲は見つかりませんでした.')

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
