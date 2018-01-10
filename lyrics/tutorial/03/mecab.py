import sys
import MeCab

m = MeCab.Tagger("^Ochasen")
print(m.parse("こんにちは、世界の皆さん。"))
