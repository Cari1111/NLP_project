import torch
import numpy as np
import pickle
from dataclasses import dataclass
from datasets import DatasetDict
from datasets.arrow_dataset import Dataset
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class ClassificationTrainer:
    model: BertForSequenceClassification
    ds: DatasetDict
    tokenizer: BertTokenizer
    identifier_str: str
    text_field: str = 'text'
    label_field: str = 'label'
    train_split: str = 'train'
    test_split: str = 'test'
    evaluation_datasets: list[tuple[str, Dataset]] = None
    model_lr: float = 1e-6
    classifier_lr: float = 1e-4

    def __post_init__(self):
        self.model.to(device)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.optimizer_network = torch.optim.Adam(self.model.bert.parameters(), lr=self.model_lr)
        self.optimizer_head = torch.optim.Adam(self.model.classifier.parameters(), lr=self.classifier_lr)
        self.losses = []
        self.evaluation_datasets = self.evaluation_datasets if self.evaluation_datasets else []
        self.evaluation_reports = {name: [] for name, _ in self.evaluation_datasets}
    
    def tokenize(self, batch):
        return self.tokenizer(list(batch[self.text_field]), truncation=True, return_tensors='pt', padding=True).to(device)
    
    def evaluate_ds(self, dataset: Dataset):
        self.model.eval()
        predictions = []
        for batch in dataset.batch(batch_size=16):
            tokenized_text = self.tokenize(batch)
            logits = self.model.forward(**tokenized_text).logits
            print(logits.shape)
            prediction = torch.argmax(logits, dim=1)
            predictions.append(prediction)
        all_predictions = torch.cat(predictions).cpu().numpy()
        self.model.train()

        print(classification_report(y_true=dataset[self.label_field], y_pred=all_predictions))
        return classification_report(y_true=dataset[self.label_field], y_pred=all_predictions, output_dict=True)
    
    def evaluate_all(self):
        for name, dataset in self.evaluation_datasets:
            report_dict = self.evaluate_ds(dataset)
            self.evaluation_reports[name].append(report_dict)
        
    def save(self):
        with open(f'models/{self.identifier_str}-losses.dat', 'wb') as f:
            pickle.dump(self.losses, f)
        with open(f'models/{self.identifier_str}-evaluations.dat', 'wb') as f:
            pickle.dump(self.evaluation_reports, f)
        self.model.save_pretrained(f'models/{self.identifier_str}-model.dat')

    def train(self, episodes, batch_size=16):
        self.evaluate_all()
        for episode in range(episodes):
            print(f'Episode {episode}')
            episode_ds = self.ds[self.train_split].shuffle()
            for step, batch in enumerate(   episode_ds.batch(batch_size=batch_size)):
                tokenized_text = self.tokenizer(list(batch[self.text_field]), truncation=True, return_tensors='pt', padding=True).to(device)
                labels = torch.tensor(batch[self.label_field]).to(device)

                self.optimizer_network.zero_grad()
                self.optimizer_head.zero_grad()

                outputs = self.model.forward(**tokenized_text, labels=labels)
                print(outputs.logits.shape)
                loss = outputs.loss

                outputs.loss.backward()
                self.optimizer_network.step()
                self.optimizer_head.step()

                print(step, loss.item(), end='\r')
                self.losses.append(loss.item())
            self.evaluate_all()
            self.save()


@dataclass
class ClassificationTrainerMultInputs:
    model: BertModel
    classifier: torch.nn.Module
    ds: DatasetDict
    tokenizer: BertTokenizer
    identifier_str: str
    text_field: str|list[str] = 'text'
    label_field: str = 'label'
    train_split: str = 'train'
    test_split: str = 'test'
    model_lr: float = 1e-6
    classifier_lr: float = 1e-4

    def __post_init__(self):
        self.model.to(device)
        self.classifier.to(device)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.optimizer_network = torch.optim.Adam(self.model.parameters(), lr=self.model_lr)
        self.optimizer_head = torch.optim.Adam(self.classifier.parameters(), lr=self.classifier_lr)
        self.losses = []
        self.evaluation_reports = []
        if type(self.text_field) is str:
            self.text_field = [self.text_field]

    def save(self):
        with open(f'models/{self.identifier_str}-losses.dat', 'wb') as f:
            pickle.dump(self.losses, f)
        with open(f'models/{self.identifier_str}-evaluations.dat', 'wb') as f:
            pickle.dump(self.evaluation_reports, f)
        #self.model.save_pretrained(f'models/{self.identifier_str}-model.dat')
    
    def forward(self, tokenized_texts):
        model_outputs = [self.model.forward(**text).pooler_output for text in tokenized_texts]
        logits = self.classifier(torch.cat(model_outputs, dim=1))
        return logits
    
    def tokenize(self, batch):
        batch_text = [list(batch[field]) for field in self.text_field]
        return [self.tokenizer(text, truncation=True, return_tensors='pt', padding=True).to(device) for text in batch_text]

    
    def evaluate(self, split):
        self.model.eval()
        ds_split = self.ds[split]
        predictions = []
        for batch in ds_split.batch(batch_size=16):
            tokenized_texts = self.tokenize(batch)
            logits = self.forward(tokenized_texts)
            prediction = torch.argmax(logits, dim=1)
            predictions.append(prediction)
        all_predictions = torch.cat(predictions).cpu().numpy()

        print(classification_report(y_true=ds_split[self.label_field], y_pred=all_predictions))
        report_dict = classification_report(y_true=ds_split[self.label_field], y_pred=all_predictions, output_dict=True)
        self.evaluation_reports.append(report_dict)
        self.model.train()


    def train(self, epochs, batch_size=16):
        self.evaluate(self.test_split)
        for epoch in range(epochs):
            print(f'--------------------- Epoch {epoch} ---------------------')
            episode_ds = self.ds[self.train_split].shuffle()
            for step, batch in enumerate(episode_ds.batch(batch_size=batch_size)):
                tokenized_texts = self.tokenize(batch)
                labels = torch.tensor(batch[self.label_field]).to(device)

                self.optimizer_network.zero_grad()
                self.optimizer_head.zero_grad()

                logits = self.forward(tokenized_texts)
                loss = self.loss_func(logits, labels)

                loss.backward()
                self.optimizer_network.step()
                self.optimizer_head.step()

                print(step, loss.item(), end='\r')
                self.losses.append(loss.item())
            
            self.evaluate(self.test_split)
            self.save()