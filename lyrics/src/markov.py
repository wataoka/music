import glob
import json
import MeCab
import random

def wakati(text):
    t = MeCab.Tagger("-Owakati")
    m = t.parse(text)
    result = m.rstrip(" \n").split(" ")

    return result



if __name__ == "__main__":

    selected_singer = ["福山雅治"]
    markov = make_markov(selected_singer)
    generate_lyric(markov)

