# MusicAIプロジェクト(仮名)

## 概要
京都大学人工知能研究会KaiRAのプロジェクト,「MusicAI(仮)」についての大まかな計画と仕様について述べる. このプロジェクトを作曲と作詞の二つに分けそれぞれのファイル名をsongとlyricsとする. 現段階では,AIによる作曲能力は少し実用的ではないため作詞のプロジェクトを進めようかと考えている. 作詞が完成した際には,それに合わせた作曲を人間により行う. そして完成した曲をプロの歌手に歌っていただき実際に販売等を行うことを目標とする. (営利性を追求しない)

## 作曲
### 作成の流れ
1. midi形式のファイルをパースする.
2. パースされたデータをLSTMに流す.
3. 出力

### 先行事例
・[RNN+LSTMで自動作曲して見た](https://qiita.com/komakomako/items/9ba38fc38f098c0e8b9b)  
(要約) RNN+LSTMをChainerで作成し,midiファイルを時系列データとして学習させた[結果](https://s3-ap-northeast-1.amazonaws.com/komahirokazu-share/rnnlstm.mp3)こうなった.

・[ニューラルネットワークで作曲をする! Magentaを動かす](https://qiita.com/marshi/items/0f6fbbe39c4381457b0a)  
(要約) MegentaというGoogleのライブラリを用いて作曲を行った. MegentaのモデルはLSTM. このモデルにmidiファイルを流して生成したところ[結果](https://soundcloud.com/ig4osq8tqokz/magenta1)こうなった。

### 雑感
- 完成したものが歌を歌うような曲ではない.
- 学習時間が長いらしい(1曲だけでも1分程度)


## 作詞
### 作成の流れ
1. データセット(既存の歌詞)をMecabで単語に切り分ける.
2. 単語をword2vecに流し,重みを学習させる.
3. そのベクトル表現から文のベクトル表現を抽出し,それをNNで学習させる.

### 今度の計画
　まずは作成の流れの1を飛ばして2を英語ベースで勉強を行う.

### 先行事例
・[RNN/LSTMを用いてさだまさし風の歌詞を自動生成してみる](https://qiita.com/moaikids/items/5c4f32d73716478fc1e1)  
(要約) deeplearning4jというjava製のOSSを使用した. このライブラリはWord2Vecだけでなく文書に応用したParagraphVectorの実装も含まれている. さだまさしの歌詞40曲をLSTMに学習させた結果,下のような歌詞ができた.

> ----- Sample 0 -----  
> 文字 指 で たどる 封 を 切っ た なら こぼれる 日射し あなた は 相 変ら ず 手紙 に まで 笑顔 同封 相 変ら ず 手紙 に まで 笑顔 で 描き やがり ふたり ふたり 初め が 間違っ て い た 君 と 出逢う 所 から だから 今日 の 日 が 来 て ひとり ひとり ひとり 僕 と 君 の 想い出 の すべて を 嘲笑う 様 に 哀しみ の 白い 影 だけ が ゆらり ゆらり ゆらり 哀しみ の 白い 影 だけ が ゆらり ゆらり ゆらり 哀しみ の 白い 影 だけ が ゆらり ゆらり ゆらりこ哀しみ の 白い 影 だけ が ゆらり ゆらり ゆらり の 訳 に は いつ
> 
> ----- Sample 1 -----  
> 文カード の おばあさん の 掃除 前 に さ り は 南車 と 言っ て た フリージア の 花束 抱い て とびだし て 来 た 君 は まるで 魔法使い 白い ドレス が とても 僕 に は まぶし すぎ て 夏 の 始まり の 小さな ふれあい私 は 心 の まま に 生き て 来 た そして その 為 に 人 を 傷つけ て 来 た 人 は それ も 知ら ず に 私 を そしり 続ける 心 の まま に 生き て ゆく の は いけ ない 事 でしょ う か 私 は あなた の 為 に 生き て 来 た そして その 為 に 人 を 傷つけ て 来 た 人 は それ も 知ら ず に
> 
> ----- Sample 2 -----  
> 文字 が とりとめ も ない ことば で 君 は どんな 気持ち で 書い た の ポスト まで かけ 出し た の 早く 着き ます 様 に って 初め みたい に 祈っ た の 季節 が かわれ ば 風邪 も なおる よ そしたら 最初 に 返事 を 書く よ くり え しきみ は 早起き し た の が さも 得意 そう に ねぼけ まなこ の 僕 を 朝食 に 追いたて ねェ また 巨人 が 負け た って ばろ 初め から そんな 二 人 じゃ なかっ た 余り 互いに 気づかい 過ぎ た の だろ う か 砂 の 上 に 二 人 で 描い た 悲しい 生活 雨 が 降る たび 風 が
> 
> ----- Sample 3 -----  
> 文A 何 度 か ら 君 の う 初め から やんな 君 を 出 ます かけ も 頃 みた いつ で も つまり の し な ず に と 届い た 手紙 ありがとう を 三 回 だけ の とても 短い 手紙 何 度 も くり返し 読み返し 乍 ら あなた の 懐しい 文字 指 で たどる 封 を 切っ た なら こぼれる 日射し あなた は 相 変ら ず 手紙 が かい ら 君 の の 出番 いいう の よう で 君 の あんま で 君 は ませ ん な 二 人 じゃ なかっ た 余り 互いに 気づかい 過ぎ た の だろ う か 砂 の 上 に 二 人 で 描い た 悲しい 生活 雨 が 降る

・[さだまさし風の歌詞を自動生成する「さだロボ」](https://qiita.com/moaikids/items/ade33723066f5bd50967)  
(要約) 上と同じ筆者が作成したもので, GUIでフレーズを自動で生成しながら追加していける. つまり, 追加するかしないかを人間が補助しながら作成する. 作者さんが気に入った歌詞が下.
> sample1
> 一度だけの手紙を書いた
> 誰かに刻んだ
> 恋はいつでも
> 僕の胸に熱が上る
> 振り返り消えそうな
> 笑顔を思い出して
  
> sample2
> 人を愛してくれた
> 君はその時に
> あなたに届くように
> 手を離さずに生きて
> いたんだね
> 
> 君の肩を抱きしめられた
> 時は過ぎて
> 恋は必ず消えて
> 
> 夢を見ていた
> 誰かの誰かの
> ために君は何処か
> 心に届くように
> あなたの夢だった  

### 雑感
２個目の記事の方法なら, 実際に楽曲を売り出す目標に現実味があるのではないかと感じた. できれば全てを人工知能が作成した曲にしたいが, 人工知能が作詞した曲ってだけでもなかなかキャッチーだし聞いてみたい十分に思うのではないだろうか.

## p.s.
音楽を作成する人工知能くんの名前を募集しています.  
(例)  
・カイラくん
