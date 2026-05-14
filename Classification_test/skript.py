import torch
import os
from datasets import load_from_disk, load_dataset
from transformers import BertForSequenceClassification, BertTokenizer

from Evaluation import Classification_Evaluation

DATASET = os.environ.get('C_DATASET')
print('DATASET:', DATASET)

MODELS = [model.split() for model in os.environ.get('C_DATASET').split('\n')]
print('MODELS:', MODELS)

NUM_HIDDEN_LAYERS = int(os.environ.get('C_NUM_HIDDEN_LAYERS'))
OUTPUT_FEATURES = int(os.environ.get('C_OUTPUT_FEATURES'))
HIDDEN_NODES = 768
CLASSIFIER = torch.nn.Sequential(
    *(torch.nn.Linear(in_features=HIDDEN_NODES, out_features=HIDDEN_NODES, bias=True) for _ in range(NUM_HIDDEN_LAYERS)),
    torch.nn.Linear(in_features=HIDDEN_NODES, out_features=OUTPUT_FEATURES, bias=True)
)

class CustomClassifierBERT(BertForSequenceClassification):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier = CLASSIFIER

dataset = load_dataset(DATASET)
tok: BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

for name, path in MODELS:
    classification_model = CustomClassifierBERT.from_pretrained(f'../Classification_training/models/{path}')

    classification_evaluation = Classification_Evaluation(
        model=classification_model,
        ds=dataset,
        tokenizer=tok,
        identifier_str=f'{DATASET}_{name}',
        label_field='label'
    )
    print('Test evaluation')
    classification_evaluation.evaluate('test')