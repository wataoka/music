import random
import MeCab
import pandas as pd

def csv2txt(filename):

    df = pd.read_csv(filename)
    lyric = df['lyric']
    lyric.to_csv('test.txt')

def wakati(text):
    t = MeCab.Tagger("-Owakati")
    m = t.parse(text)
    result = m.rstrip(" \n").split(" ")
    return result

if __name__ == "__main__":
    csv2txt('./data/01/#嘲笑ポラロイド.csv')
    filename = "test.txt"
    src = open(filename ,"r").read()
    wordlist = wakati(src)

    # マルコフ連鎖用のテーブルを作成する
    markov = {}
    w1 = ""
    w2 = ""
    for word in wordlist:
        if w1 and w2:
            if (w1, w2) not in markov:
                markov[(w1, w2)] = []
            markov[(w1, w2)].append(word)
        w1, w2 = w2 ,word


    count = 0
    sentence = ""
    w1, w2 = random.choice(list(markov.keys()))
    while count < len(wordlist):
        tmp = random.choice(markov[(w1, w2)])
        sentence += tmp
        w1, w2 = w2, tmp
        count += 1

    print(sentence)
