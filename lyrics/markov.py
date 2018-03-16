import random
import MeCab
import glob


def wakati(text):

    t = MeCab.Tagger("-Owakati")
    m = t.parse(text)
    result = m.rstrip(" \n").split(" ")
    return result



def markov():

    file_list = glob.glob('./data/txt/*.txt')
    markov = {}
    for filename in file_list:
        src = open(filename, 'r').read()
        wordlist = wakati(src)
        w1 = ""
        w2 = ""
        for word in wordlist:
            if w1 and w2:
                if(w1, w2) not in markov:
                    markov[(w1, w2)] = []
                markov[(w1, w2)].append(word)
            w1, w2 = w2, word

    count = 0
    sentence = ""
    w1, w2 = random.choice(list(markov.keys()))
    while count < 300:
        tmp = random.choice(markov[(w1, w2)])
        sentence += tmp
        w1, w2 = w2, tmp
        count += 1

    print(sentence)



if __name__ == "__main__":
    markov()
    # filename = "./data/txt/#嘲笑ポラロイド.txt"
    # src = open(filename ,"r").read()
    # wordlist = wakati(src)

    # # マルコフ連鎖用のテーブルを作成する
    # markov = {}
    # w1 = ""
    # w2 = ""
    # for word in wordlist:
    #     if w1 and w2:
    #         if (w1, w2) not in markov:
    #             markov[(w1, w2)] = []
    #         markov[(w1, w2)].append(word)
    #     w1, w2 = w2 ,word


    # count = 0
    # sentence = ""
    # w1, w2 = random.choice(list(markov.keys()))
    # while count < len(wordlist):
    #     tmp = random.choice(markov[(w1, w2)])
    #     sentence += tmp
    #     w1, w2 = w2, tmp
    #     count += 1

    # print(sentence)
