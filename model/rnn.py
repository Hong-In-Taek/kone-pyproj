'''
import os
from pprint import pprint
import pandas as pd
import numpy as np
import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
from konlpy.tag import Okt
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def get_word2vec(model_path):

    w2v = gensim.models.Word2Vec.load(model_path)
    w2v_weights = w2v.trainables.syn1neg
    w2v_weights = np.concatenate(
        [w2v_weights, np.random.randn(1, 200).astype(np.float32)])  # pad 토큰 임베딩 추가
    w2v_weights = torch.FloatTensor(w2v_weights)
    id2word = w2v.wv.index2word

    print(f"* 전체 단어 사전 크기 : {w2v_weights.shape[0]} 개")
    print(f"* 전체 Word Embedding 크기 : {w2v_weights.shape}")
    return w2v, w2v_weights, id2word


w2v, w2v_weights, id2word = get_word2vec("static/models/ko.bin")


def make_padding(data, reverse=True):

    max_sen_len = 0
    for to in data.text:
        if max_sen_len < len(to):
            max_sen_len = len(to)

    print(f"* 토큰화 된 문장의 최대 길이 : {max_sen_len}\n")

    for i, sentence in enumerate(tqdm(data.text, desc="Padding the data..")):
        pad_len = max_sen_len - len(sentence)
        pads = [30185] * pad_len
        sentence.extend(pads)
        if reverse:
            sentence.reverse()  # 원활한 gradient flow를 위한 input reverse (안녕 하세요 -> 하세요 안녕)
    print("\n")
    print("done.")
    return max_sen_len, data


sequence_length = 50
vocab_size = w2v_weights.shape[0]
input_size = w2v_weights.shape[1]
hidden_size = 200
num_layers = 2
num_classes = 2
num_epochs = 10
learning_rate = 5e-4      # For RNN


def tokenizing(tokenizer, data):
    tokenized_data = data.copy()

    for i, sentence in enumerate(tqdm(data.text, desc="Tokening the data...")):
        tokenized = tokenizer.morphs(sentence)
        tokenized_data.text[i] = tokenized

    print("\n")
    print("done.")
    return tokenized_data


def token_indexing(data):
    input_data = data.copy()

    for i, sentence in enumerate(tqdm(data.text, desc="Indexing the tokenized data to vocab index...")):
        tmp = []
        for token in sentence:
            if token in id2word:
                tmp.append(id2word.index(token))

    input_data.text[i] = tmp
    print("\n")
    print("done.")
    return input_data


# For Web Crawling
class Test_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

        self.x = self.data['text']
        self.x = self.x.values

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.IntTensor(self.x[idx])

        return x


class Trainer:
    def __init__(self, model, learning_rate, num_epochs, device, save_dir):
        self.model = model
        self.model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate)

        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        self.device = device
        self.save_dir = save_dir

        self.best_acc = 0
        self.score_list = []

    def test_eval(self, test_loader):
        self.model.eval()
        pos = 0
        neg = 0
        test_data_size = 0
        with torch.no_grad():
            for input in test_loader:
                input = input.to(self.device)
                outputs = self.model(input)
                recommendation = torch.round(outputs.softmax(dim=-1)[:, 1]*10)
                test_data_size += len(recommendation)
                pos += (recommendation >= 6).sum()
                neg += (recommendation < 2).sum()

            nps = (100 * pos/test_data_size) - (100 * neg/test_data_size)
            if nps < 0:
                nps = 0
            self.score_list.append(int(nps))

            # 앱 추천 지수 출력
            print(f'Recommendation score: {nps}')
            print('---------------------------------------------')


# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, num_layers, num_classes, device):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        # random embedding
        self.embedding = nn.Embedding(vocab_size, input_size)

        self.rnn = nn.RNN(input_size, hidden_size,
                          num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def from_pretrained(self, weights):
        self.embedding = nn.Embedding.from_pretrained(weights)
        print("Successfully load the pre-trained embeddings.")

    def forward(self, x):
        x = self.embedding(x)

        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(self.device)

        # Forward propagate rnn
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.rnn(x, h0)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return out


def analyze_test_data(sentences):
    tokenizer = Okt()
    trainer = Trainer()
    test_data = pd.DataFrame({'text': [sentences]})

  #  print(f'{test_data.app[0]}')
   # app_names.append(test_data.app[0])
    # test_data = test_data.drop(['app'], axis=1)

    # tokenizing
    test_data = tokenizing(tokenizer, test_data)

    # Get the pre-trained word2vec model and its weights.
    w2v, w2v_weights, id2word = get_word2vec("static/models/ko.bin")

    # Word indexing
    test_data = token_indexing(test_data)

    # Make Padding
    max_sen_len, test_data = make_padding(test_data)

    print('\nPreprocessing Done !!!')

    # Prepare the test data
    test_dataset = Test_Dataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    # Do test
    trainer.test_eval(test_loader)

#    df = pd.DataFrame([x for x in zip("app_names", trainer.score_list)])
#    df.columns = ['Name', 'Recommendation Score']
    return trainer.score_list
'''
