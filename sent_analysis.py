import pandas as pd
import re

import torch

from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification


#define dataset class that returns input ids and attention mask of 
#tokenized input sequences alongside with target labels
class SentimentAnalysisDatasetDistilBert(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) 
            for key, val in self.encodings.items()
        }
        item['labels'] = torch.tensor(self.labels[idx])

        return item
    
    def __len__(self):
        return len(self.labels)


#initial preprocessing of the data
def clean_text(text):
    text = text.lower() 
    text = re.sub('http\S+|www.\S+', '', text)
    text = re.sub('@[^\s]+', '', text)
    text = re.sub(r'#([^\s]+)', '', text)    
    text = re.sub('\[.*?\]', '', text) 
    text = re.sub('\s+', ' ', text).strip()
    text = re.sub(r'\w*\d\w*', '', text)


    return text

#load and preprocess
train_data = pd.read_csv('train.csv')

train_data = train_data.loc[:, ['text', 'sentiment']]

train_data = train_data.dropna()

train_data['text'] = train_data['text'].apply(clean_text)

sent_mappings = {'negative':0, 'neutral':1, 'positive':2}
train_data['sentiment'] = train_data['sentiment'].map(lambda x: sent_mappings[x])

#tokenize
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
encoded = tokenizer(list(train_data['text'].values), truncation=True, padding=True)

#create dataset object and corresponding loader
train_dataset = SentimentAnalysisDatasetDistilBert(encoded, train_data['sentiment'].values)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

#create model and Adam optimizer
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
optim = torch.optim.Adam(model.parameters(), lr=0.001)

#mini-batch training loop
for epoch in range(100):
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = model(input_ids, attention_mask)

        logits = outputs['logits']
       
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if batch_idx % 100 == 0:
            print(f"epoch {epoch}: batch {batch_idx}, loss = {loss:.6f}")