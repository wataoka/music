import sys
import MeCab

mecab = MeCab.Tagger("-Ochasen -d /usr/lib/mecab/dic/mecab-ipadic-neologd")
print(mecab.parse("こんにちは、世界のみなさん。"))
