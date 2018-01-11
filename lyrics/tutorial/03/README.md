# 作詞チュートリアル03
## 概要
作詞チュートリアル03では, mecabについて学ぶ.

## インストール
```command
$ pip install mecab-python3
```
でOK.

## MeCabとは
　日本語を単語に分割し, それぞれの単語に品詞を付与することを形態素解析という. そしてMeCabは形態素解析エンジン.

## 色々使ってみよう
### ■その1
```python
import sys
import MeCab

mecab = MeCab.Tagger("-Ochasen")
print(mecab.parse("こんにちは、世界のみなさん。"))
```
```command
こんにちは	感動詞,*,*,*,*,*,こんにちは,コンニチハ,コンニチワ
、	記号,読点,*,*,*,*,、,、,、
世界	名詞,一般,*,*,*,*,世界,セカイ,セカイ
の	助詞,連体化,*,*,*,*,の,ノ,ノ
皆さん	名詞,一般,*,*,*,*,皆さん,ミナサン,ミナサン
。	記号,句点,*,*,*,*,。,。,。
EOS
```

### ■その2
```python
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

```
```command
['私', '名前', '綿', '岡', '晃', '輝', '生まれ', '大阪', 'この世', '好き', 'もの', '数学']
```
