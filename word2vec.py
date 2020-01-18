import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gensim
from collections import defaultdict
from sklearn.utils import shuffle


class LSTMClassifier(nn.Module):
    # モデルで使う各ネットワークをコンストラクタで定義
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, weights, enable_fine_tuning=True):
        # 親クラスのコンストラクタ, 決まり文句
        super(LSTMClassifier, self).__init__()
        # 隠れ層の次元数
        self.hidden_dim = hidden_dim
        # 単語埋め込み層の定義, inputの単語をベクトル化するために使う
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        weights = torch.from_numpy(weights[:vocab_size])
        pad = torch.zeros(1, weights.size(1))  # 零ベクトル
        unk = torch.from_numpy(np.random.uniform(low=-0.05, high=0.05, size=(1, weights.size(1))).astype(np.float32))  # 未知語
        sepecial_token = torch.cat([pad, unk], dim=0)  # padとunkをconcatする
        self.word_embeddings.weight = nn.Parameter(torch.cat([sepecial_token, weights], dim=0))
        self.word_embeddings.requires_grad = enable_fine_tuning
        # LSTMの隠れ層
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # LSTMの出力を受け取って全結合してsoftmaxに食わせるための1層のネットワーク
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        # softmaxのLog版
        self.softmax = nn.LogSoftmax()

    # 順伝播処理, forward関数に記載
    def forward(self, sentence):  # sentenceは単語のindexのリスト
        # 文の各単語のindexからなるテンソルをベクトルに変換, 二次元のテンソル
        embeds = self.word_embeddings(sentence)
        # many to oneのタスクを解きたいので、第二戻り値だけを使う
        _, lstm_out = self.lstm(embeds)  # 単語のベクトルのテンソルをlstmに流し込む
        # targetの空間に写像する
        tag_space = self.hidden2tag(lstm_out[0])
        # softmaxに食わせて確率として表現
        tag_scores = self.softmax(tag_space.squeeze())
        return tag_scores


# 単語ID辞書を作成する
def create_word2index(vocab):
    word2index = defaultdict(lambda: len(word2index))
    # 定数なのでグローバルにアクセス (非推奨)
    word2index[PADDIGN_WORD]  # 0がセットされる
    word2index[UNKNOWN_WORD]  # 1がセットされる
    for word in vocab[:VOCAB_SIZE]:  # 定数なのでグローバルにアクセス
        word2index[word]

    # 登録されていない単語にアクセスされたときに新しいidが付与されないようにdictに変換しておく
    return dict(word2index)


# 文章を単語IDの系列データに変換
# PyTorchのLSTMのインプットとなるデータなのでtensor型
def sentence2index(sentence):
    words = sentence.split()
    # 未知語はUNKNOWN_WORDに対応したidである'1'に変換
    return torch.tensor([word2index[w] if w in word2index else word2index[UNKNOWN_WORD] for w in words], dtype=torch.long)


def category2tensor(cat):
    return torch.tensor([category2index[cat]], dtype=torch.long)


# データをバッチサイズ毎に分割する関数
def train2batch(sentences, labels, batch_size=128):
    sentences_batch = []
    labels_batch = []
    sentences_shuffle, labels_shuffle = shuffle(sentences, labels)
    for i in range(0, len(sentences_shuffle), batch_size):
        sentences_batch.append(sentences_shuffle[i:i+batch_size])
        labels_batch.append(labels_shuffle[i:i+batch_size])
    return sentences_batch, labels_batch


# 'word2vecモデル読み込み' or 'fasttextモデル読み込み'
model_dir = "./model/GoogleNews-vectors-negative300.bin"
# model_dir = "./model/fasttext.en.bin"
embedding_model = gensim.models.KeyedVectors.load_word2vec_format(model_dir, binary=True, limit=30000)
word_vectors = embedding_model.wv
weights = word_vectors.syn0

# 単語のベクトル次元数
EMBEDDING_DIM = weights.shape[1]  # 300
# 隠れ層の次元数
HIDDEN_DIM = 128
# データ全体の単語数
VOCAB_SIZE = weights.shape[0]  # 30000
# 分類先のカテゴリの数
TAG_SIZE = 2  # 二値分類
# padding, 未知語
PADDIGN_WORD = '<PAD>'
UNKNOWN_WORD = '<UNK>'

# カテゴリ辞書を作成する, 二値分類
category2index = {"0": 0, "1": 1}

# 単語ID辞書を作成する
word2index = create_word2index(list(embedding_model.wv.vocab.keys()))

# モデル宣言
model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, TAG_SIZE, weights)

GPU = True
device = torch.device("cuda" if GPU else "cpu")
model = model.to(device)

# 損失関数はNLLLoss()を使う
loss_function = nn.NLLLoss()
# 最適化の手法はSGD
optimizer = optim.SGD(model.parameters(), lr=0.01)

# データセット読み込み
trainset = pd.DataFrame(columns=["sentence", "label"])
with open("./data/train.tsv", "r") as f:
    lines = f.readlines()
    for line in lines:
        sentence, label = line.split("\t")
        label = label.rstrip("\n")
        s = pd.Series([sentence, label], index=trainset.columns)
        trainset = trainset.append(s, ignore_index=True)

devset = pd.DataFrame(columns=["sentence", "label"])
with open("./data/dev.tsv", "r") as g:
    lines = g.readlines()
    for line in lines:
        sentence, label = line.split("\t")
        label = label.rstrip("\n")
        s = pd.Series([sentence, label], index=devset.columns)
        devset = devset.append(s, ignore_index=True)

# データフレームシャッフル
trainset = trainset.sample(frac=1).reset_index(drop=True)
devset = devset.sample(frac=1).reset_index(drop=True)

trainset_sentences_tmp = []  # ひとまずtmp, paddingを埋める前の文を格納するリスト
trainset_labels = []

# 系列の最大長を取得
max_len = 0
for sentence, label in zip(trainset["sentence"], trainset["label"]):
    trainset_sentences_tmp.append(sentence2index(sentence))  # テンソルに変換して追加
    trainset_labels.append(category2index[label])
    if max_len < len(sentence.split()):
        max_len = len(sentence.split())
# 系列の長さを揃えるために短い系列にpaddingを追加
trainset_sentences = []
for sentence_index in trainset_sentences_tmp:  # sentence_indexはテンソル
    pad = torch.tensor([0]*(max_len-len(sentence_index)))
    if max_len - len(sentence_index) > 0:
        sentence_index = torch.cat([sentence_index, pad])  # 後ろに0を追加
    trainset_sentences.append(sentence_index)

# 学習を開始する
losses = []
# 100エポック

for epoch in range(100):
    model.train()
    all_loss = 0
    mini_batch_sentences, mini_batch_labels = train2batch(trainset_sentences, trainset_labels)
    for i in range(len(mini_batch_sentences)):
        batch_loss = 0
        model.zero_grad()

        # 順伝播させるtensorはGPUで処理させるためdevice=GPUをセット, 128文ずつ学習する
        sentence_tensor = torch.stack(mini_batch_sentences[i]).cuda(device=device)
        label_tensor = torch.tensor(mini_batch_labels[i], device=device).squeeze()
        # 順伝播の結果を受け取る
        out = model(sentence_tensor)
        # 正解とのlossを計算
        batch_loss = loss_function(out, label_tensor)
        # 勾配をセット
        batch_loss.backward()
        # 逆伝播でパラメータ更新
        optimizer.step()
        # lossを計算
        all_loss += batch_loss.item()
    losses.append(all_loss)
    print(f"epoch{epoch}: loss {all_loss}")
    model.eval()
    with torch.no_grad():
        pass
print("done")

# ひとまず完成, 今後の予定
# エポックが終わる毎にdevセットの予測を行う
# devセットのロスが下がったタイミングでモデルのパラメータを保存
torch.save(model.state_dict(), f"./model/{epoch}_w2v_lstm.pth")
