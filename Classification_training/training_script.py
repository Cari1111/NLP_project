import torch
import os
from datasets import load_from_disk
from transformers import BertTokenizer, BertModel, BertForSequenceClassification

from Trainer import ClassificationTrainer


print('======================================================')

IDENTIFIER_STR = os.environ["C_IDENTIFIER_STR"]
print('IDENTIFIER_STR:', IDENTIFIER_STR)

DATASET = os.environ.get('C_DATASET')
print('DATASET:', DATASET)

EVALUATION_DATASETS = [model.split() for model in os.environ.get('C_EVALUATION_DATASETS').split('\n') if len(model)>0]
print('EVALUATION_DATASETS:', EVALUATION_DATASETS)

TEXT_FIELD = os.environ.get('C_TEXT_FIELD')
print('TEXT_FIELD:', TEXT_FIELD)

LABEL_FIELD = os.environ.get('C_LABEL_FIELD')
print('LABEL_FIELD:', LABEL_FIELD)

EPOCHS = int(os.environ.get('C_EPOCHS'))
print('EPOCHS:', EPOCHS)

LR_MODEL = int(os.environ.get('C_LR_MODEL'))
print('LR_MODEL:', LR_MODEL)

LR_CLASSIFIER = int(os.environ.get('C_LR_CLASSIFIER'))
print('LR_CLASSIFIER:', LR_CLASSIFIER)

NUM_HIDDEN_LAYERS = int(os.environ.get('C_NUM_HIDDEN_LAYERS'))
OUTPUT_FEATURES = int(os.environ.get('C_OUTPUT_FEATURES'))
HIDDEN_NODES = 768*len(TEXT_FIELD)
CLASSIFIER = torch.nn.Sequential(
    *(torch.nn.Linear(in_features=HIDDEN_NODES, out_features=HIDDEN_NODES, bias=True) for _ in range(NUM_HIDDEN_LAYERS)),
    torch.nn.Linear(in_features=HIDDEN_NODES, out_features=OUTPUT_FEATURES, bias=True)
)
print('CLASSIFIER:', CLASSIFIER)

MODELS = [model.split() for model in os.environ.get('C_MODELS').split('\n') if len(model)>0]
print('MODELS:', MODELS)

print('======================================================')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
print('Device:', device)

dataset = load_from_disk(DATASET)
print(dataset, dataset['train'][0])

evaluation_datasets = [(name, load_from_disk(path)[split]) for name, split, path in EVALUATION_DATASETS]
evaluation_datasets.append(('test_split', dataset['test']))
print(evaluation_datasets)
for name, d in evaluation_datasets:
    print(name, set(d['label']))
print('train_split', set(dataset['train']['label']))

tok: BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

for model, path in MODELS:
    print('======================================================')
    print(model, path)
    id_str = IDENTIFIER_STR.replace('MODEL', model)
    print(id_str)
    classification_model = BertForSequenceClassification.from_pretrained(path)
    print('======================================================')

    classification_trainer = ClassificationTrainer(
        model=classification_model,
        ds=dataset,
        tokenizer=tok,
        identifier_str=id_str,
        text_field=TEXT_FIELD,
        label_field=LABEL_FIELD,
        evaluation_datasets=evaluation_datasets,
        model_lr=10**(-LR_MODEL) if LR_MODEL != 0 else 0,
        classifier_lr=10**(-LR_CLASSIFIER)
    )

    classification_trainer.train(EPOCHS, batch_size=32)

    print('======================================================')

# classification_evaluation = Classification_Evaluation(
#     model=classification_model,
#     ds=dataset,
#     tokenizer=tok,
#     identifier_str=IDENTIFIER_STR,
#     label_field=LABEL_FIELD
# )
# print('Train evaluation')
# classification_evaluation.evaluate('train')
# print('Test evaluation')
# classification_evaluation.evaluate('test')