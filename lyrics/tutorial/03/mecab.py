import sys
import MeCab


mecab = MeCab.Tagger("mecabrc")

def ma_parse(sentence, filter="名詞"):
    node = mecab.parseToNode(sentence)
    while node:
        if node.feature.startswith(filter):
            yield node.surface
        node = node.next

if __name__ == "__main__":
    sentence = "私の名前は綿岡晃輝です。生まれは大阪。この世で最も好きなものは数学です。"
    words = [word for word in ma_parse(sentence)]

    print(words)
