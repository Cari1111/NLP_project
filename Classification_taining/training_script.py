import os, math, random, torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional
from datasets import load_dataset, Dataset
from transformers import BertTokenizer, BertForMaskedLM, BertForSequenceClassification
from torch.utils.data import default_collate

from Trainer import ClassificationTrainer
from Evaluation import Classification_Evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
print('Device:', device)

yelp_dataset = load_dataset("datasets/yelp_ds_preprocessed")

ckpt = "bert-base-uncased"     # swap for domain/multilingual BERT as needed
tok: BertTokenizer = BertTokenizer.from_pretrained(ckpt)
classification_model = BertForSequenceClassification.from_pretrained(ckpt)

classification_trainer = ClassificationTrainer(
    model=classification_model,
    ds=yelp_dataset['train'],
    tokenizer=tok,
)

classification_trainer.train(4, batch_size=32)

classification_evaluation = Classification_Evaluation(
    model=classification_model,
    ds=yelp_dataset,
    tokenizer=tok,
)
print('Train evaluation')
classification_evaluation.evaluate('train')
print('Test evaluation')
classification_evaluation.evaluate('test')