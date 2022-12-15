import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertForSequenceClassification as BertModel


tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
class Test_Dataset(Dataset):
    
    def __init__(self, data):
      self.data = data

      self.x = self.data['text'].values

    def __len__(self):
      return len(self.x)

    def __getitem__(self, idx):
        
      x = tokenizer(self.x[idx], padding='max_length', truncation=True, return_tensors="pt", max_length=128)

      return x



def Analyze_data_sentencetest(sentense, finetuned=True):
    # Preprocessing
    pos = 0
    neg = 0

    test_data = pd.DataFrame({'text': [sentense]})
    device = torch.device('cpu')
    # Evaluation
    torch.cuda.empty_cache()
    test_dataset = Test_Dataset(test_data)
    print(test_data)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = BertModel.from_pretrained('skt/kobert-base-v1')
        ## 미세조정을 거친 모델을 사용할 것인지,
        ## 거치지 않은 모델을 사용할 것인지.

    model.load_state_dict(torch.load('/content/drive/MyDrive/Colab Notebooks/koreaUniv/실습/실습#12/model/best_koberttmp.pt'))
    

    model.eval()
    sizerecon=0
    with torch.no_grad():
        for input in test_loader:
            ids = input['input_ids'].view(batch_size, -1) # 100, 1, 128 => 100, 128
            ids = ids.to(device)
            att_mask = input['attention_mask'].view(batch_size, -1)
            att_mask = att_mask.to(device)
                
            with torch.no_grad():
                output = model(ids, att_mask)
                
            recommendation = torch.round(output.logits.softmax(dim=-1)[:, 1]*10)
            sizerecon += len(recommendation)
            print(recommendation)
            pos += (recommendation > 8).sum()
            neg += (recommendation <= 8).sum()
            print(pos,neg)
            print(sizerecon)
            nps = (100 * pos / sizerecon)  - (100 * neg / sizerecon) 

        if nps < 0:
            nps = 0

    print('\nRecommendation score: {} %'.format(nps)) 
    print('---------------------------------------------')

    return ""