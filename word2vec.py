# 必要なライブラリのimport
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gensim
from collections import defaultdict


class LSTMClassifier(nn.Module):
    # モデルで使う各ネットワークをコンストラクタで定義
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        # 親クラスのコンストラクタ。決まり文句
        super(LSTMClassifier, self).__init__()
        # 隠れ層の次元数
        self.hiddne_dim = hidden_dim
        # インプットの単語をベクトル化するために使う
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # LSTMの隠れ層
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        # LSTMの出力を受け取って全結合してsoftmaxに食わせるための1層のネットワーク
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        # softmaxのLog版. dim=0で列, dim=1で行方向を確率変換
        self.softmax = nn.LogSoftmax(dim=1)

    def set_pretrained(self, weights, enable_fine_tuning=True):
        vocab_size = self.word_embeddings.weight.shape[0]
        # ここでcopyするのは、のちにvocab_size次元目をランダムな値で初期化するため
        # numpyのsliceはシャロウコピー（リストに対するsliceはディープコピー）なので、sliceするにしてもcopyしておかないと元の値が変わってしまう
        weights = weights[:vocab_size]
        weights = weights.copy()
        # ランダムな値で初期化
        weights[-1] = np.random.randn(EMBEDDING_DIM)

        self.word_embeddings = nn.Embedding.from_pretrained(torch.tensor(weights))

    # 順伝播処理はforward関数に記載
    def forward(self, sentence):
        # 文章内の各単語をベクトル化して出力. 二次元のテンソル
        embeds = self.word_embeddings(sentence)
        # 二次元テンソルをLSTMに食わせられる様にviewで三次元テンソルにした上でLSTMへ流す.
        # 上記で説明した様にmany to oneのタスクを解きたいので、第二戻り値だけを使う。
        _, lstm_out = self.lstm(embeds.view(len(sentence), 1, -1))
        # lstm_out[0]は三次元テンソルになってしまっているので二次元に調整して全結合
        tag_space = self.hidden2tag(lstm_out[0].view(-1, HIDDEN_DIM))
        # softmaxに食わせて確率として表現
        tag_scores = self.softmax(tag_space)
        return tag_scores


# 単語ID辞書を作成する
def create_word2index(vocab):
    word2index = defaultdict(lambda: len(word2index))
    # unknownを入れるためにVOCAB_SIZE-1までで止めておく
    for word in vocab[:VOCAB_SIZE-1]:
        word2index[word]

    # 定数なのでグローバルにアクセス（やや横着）
    word2index[UNKNOWN_WORD]

    # 新しい単語が追加された時に新しいidが付与されないようdictに変換しておく
    return dict(word2index)


# 文章を単語IDの系列データに変換
# PytorchのLSTMのインプットになるデータなので、もちろんtensor型で
def sentence2index(sentence):
    words = sentence.split()
    # 未知語はUNKNOWN_WORDに対応したidに変換
    return torch.tensor([word2index[w] if w in word2index else word2index[UNKNOWN_WORD] for w in words], dtype=torch.long)


def category2tensor(cat):
    return torch.tensor([category2index[cat]], dtype=torch.long)


# word2vecモデル読み込み等
model_dir = "./GoogleNews-vectors-negative300.bin"
embedding_model = gensim.models.KeyedVectors.load_word2vec_format(model_dir, binary=True)
word_vectors = embedding_model.wv
weights = word_vectors.syn0

# 単語のベクトル次元数
EMBEDDING_DIM = weights.shape[1]
# 隠れ層の次元数
HIDDEN_DIM = 128
# データ全体の単語数
VOCAB_SIZE = 30000
# 分類先のカテゴリの数
TAG_SIZE = 2  # ニ値分類
# 未知語
UNKNOWN_WORD = '__UNK__'

# カテゴリ辞書を作成する
category2index = {}
for cat in range(2):  # 0か1
    category2index[cat] = len(category2index)

# 単語ID辞書を作成する
# スライスするのでlist()としている
word2index = create_word2index(list(embedding_model.wv.vocab.keys()))

# モデル宣言
model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, TAG_SIZE)
model.set_pretrained(weights)

# GPU = True
# device = torch.device("cuda" if GPU else "cpu")
# model = model.to(device)

# 損失関数はNLLLoss()を使う, LogSoftmaxを使う時はこれを使うらしい.
loss_function = nn.NLLLoss()
# 最適化の手法はSGDで. lossの減りに時間かかるけど、一旦これを使う.
optimizer = optim.SGD(model.parameters(), lr=0.01)

# データセット読み込み
datasets = pd.DataFrame(columns=["sentence", "label"])

with open("./data/train.tsv", "r") as f:
    lines = f.readlines()
    for line in lines:
        sentence, label = line.split("\t")
        label = label.rstrip()
        s = pd.Series([sentence, label], index=datasets.columns)
        datasets = datasets.append(s, ignore_index=True)

# データフレームシャッフル
datasets = datasets.sample(frac=1).reset_index(drop=True)


# 学習を開始する
# 各エポックの合計のloss値を格納する
losses = []
# 100回ループ回してみる。
for epoch in range(30):
    all_loss = 0
    for sentence, label in zip(datasets["sentence"], datasets["label"]):
        # モデルが持っている勾配の情報をリセット
        model.zero_grad()
        # 文章を単語IDの系列に変換 (modelに食わせられる形に変換)
        inputs = sentence2index(sentence)
        # 順伝播の結果を受け取る
        out = model.forward(inputs)
        # 正解カテゴリをテンソル化
        # ここはワイが修正
        answer = category2tensor(int(label))
        # 正解とのlossを計算
        loss = loss_function(out, answer)
        # 勾配をセット
        loss.backward()
        # 逆伝播でパラメータ更新
        optimizer.step()
        # lossを計算
        all_loss += loss.item()
    losses.append(all_loss)
    print("epoch", epoch, "\t", "loss", all_loss)
print("done")
# モデルの保存はこれから行う
