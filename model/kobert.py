
import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook


from kobert_tokenizer import KoBERTTokenizer

from transformers import BertForSequenceClassification as BertModel
#tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

batch_size = 16
sequence_length = 128
num_epochs = 5
learning_rate = 1e-5


class Test_Dataset(Dataset):

    def __init__(self, data):
        self.data = data

        self.x = self.data['text'].values

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = tokenizer(self.x[idx], padding='max_length', truncation=True,
                      return_tensors="pt", max_length=sequence_length)
        return x


class kobertMbti:

    model = BertModel.from_pretrained('skt/kobert-base-v1')

    def analyze_sentence(self, sentence, kind, finetuned=True):
        # Preprocessing
        test_data = pd.DataFrame([sentence], columns=['text'])
        print(test_data)
        # Evaluation
      #  device = torch.device('cuda')
      #  torch.cuda.empty_cache()
        test_dataset = Test_Dataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        if finetuned:
            if kind == "1":
                self.model.load_state_dict(torch.load(
                    'static/models/best_kobert_1.pt', map_location=torch.device('cpu')))
            elif kind == "2":
                self.model.load_state_dict(torch.load(
                    'static/models/best_kobert_2.pt', map_location=torch.device('cpu')))
            elif kind == "3":
                self.model.load_state_dict(torch.load(
                    'static/models/best_kobert_3.pt', map_location=torch.device('cpu')))
            elif kind == "4":
                self.model.load_state_dict(torch.load(
                    'static/models/best_kobert_4.pt', map_location=torch.device('cpu')))
            else:
                self.model.load_state_dict(torch.load(
                    'static/models/best_kobert.pt_1.pt', map_location=torch.device('cpu')))
       # self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            for input in test_loader:

                ids = input['input_ids'].view(
                    batch_size, -1)  # 100, 1, 128 => 100, 128
         #       ids = ids.to(device)
                att_mask = input['attention_mask'].view(batch_size, -1)
          #      att_mask = att_mask.to(device)

                with torch.no_grad():
                    output = self.model(ids, att_mask)

                    recommendation = torch.round(
                        output.logits.softmax(dim=-1)[:, 1] * 10).sum().item()

                    # 앱 추천 지수 출력
                    # print('Recommendation score: {} %'.format(predicted))
                    print('Recommendation score: {} %'.format(recommendation))
                    print('---------------------------------------------')
                    return recommendation

    def mbtiFuction(self, sentence):
        mbti = ""
        ei = self.analyze_sentence(sentence, "1", finetuned=True)
        ns = self.analyze_sentence(sentence, "2", finetuned=True)
        tf = self.analyze_sentence(sentence, "3", finetuned=True)
        pj = self.analyze_sentence(sentence, "4", finetuned=True)
        if (float(ei) > 60.0):
            mbti += "e"
        else:
            mbti += "i"

        if (float(ns) > 60.0):
            mbti += "n"
        else:
            mbti += "s"

        if (float(tf) > 60.0):
            mbti += "t"
        else:
            mbti += "f"

        if (float(pj) > 60.0):
            mbti += "p"
        else:
            mbti += "j"

        return mbti
