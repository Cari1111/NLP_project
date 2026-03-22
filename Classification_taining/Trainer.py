import torch
import numpy as np
import pickle
from dataclasses import dataclass
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class ClassificationTrainer:
    model: BertForSequenceClassification
    ds: Dataset
    tokenizer: BertTokenizer
    identifier_str: str
    text_field: str = 'text'
    label_field: str = 'label'

    def __post_init__(self):
        self.model.to(device)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.optimizer_network = torch.optim.Adam(self.model.bert.parameters(), lr=1e-6)
        self.optimizer_head = torch.optim.Adam(self.model.classifier.parameters(), lr=1e-4)
        self.losses = []

    def save(self):
        with open(f'models/{self.identifier_str}-losses.dat', 'wb') as f:
            pickle.dump(self.losses, f)
        self.model.save_pretrained(f'models/{self.identifier_str}-model.dat')

    def train(self, episodes, batch_size=16):
        for episode in range(episodes):
            print(f'Episode {episode}')
            self.ds.shuffle()
            for step, batch in enumerate(self.ds.batch(batch_size=batch_size)):
                tokenized_text = self.tokenizer(list(batch[self.text_field]), truncation=True, return_tensors='pt', padding=True).to(device)
                labels = torch.tensor(batch[self.label_field]).to(device)

                self.optimizer_network.zero_grad()
                self.optimizer_head.zero_grad()

                outputs = self.model(**tokenized_text, labels=labels)
                loss = outputs.loss

                outputs.loss.backward()
                self.optimizer_network.step()
                self.optimizer_head.step()

                print(step, loss.item(), end='\r')
                self.losses.append(loss.item())
            self.save()