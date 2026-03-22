import torch
from datasets import load_from_disk
from transformers import BertTokenizer, BertForSequenceClassification

from Trainer import ClassificationTrainer
from Evaluation import Classification_Evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
print('Device:', device)

yelp_dataset = load_from_disk("datasets/yelp_ds_preprocessed")
print(yelp_dataset, yelp_dataset['train'][0])

ckpt = "bert-base-uncased"     # swap for domain/multilingual BERT as needed
tok: BertTokenizer = BertTokenizer.from_pretrained(ckpt)
classification_model = BertForSequenceClassification.from_pretrained(ckpt)

classification_trainer = ClassificationTrainer(
    model=classification_model,
    ds=yelp_dataset['train'],
    tokenizer=tok,
    identifier_str='yelp_4eps_1'
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