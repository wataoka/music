import MeCab
import re

text = "私の名前はカイラです。今年もよろしくお願いします。"

# 分かち書き
mecab = MeCab.Tagger("-Owakati")
result = mecab.parse(text)

# 単語リスト作成
# 正規表現を用いてスペースで分けている
ws = re.compile(" ")
words = [word for word in ws.split(result)]
if words[-1] == u'\n':
    words = words[:-1]

print(words)
