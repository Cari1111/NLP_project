import torch
from datasets import load_from_disk
from transformers import BertTokenizer, BertForSequenceClassification

from Trainer import ClassificationTrainer
from Evaluation import Classification_Evaluation

# -----------------------------------------------

IDENTIFIER_STR = 'yelp_4eps_1-losses' #0-2_CoT_4eps_1'

DATASET = "Scientific-text-classification-preprocessed" #"yelp_ds_preprocessed"
LABEL_FIELD = 'label_idx'

MODEL_OUT = 10

WORK_DIR = '/mnt/lustre/work/eickhoff/esx833/NLP_project/'
MODEL_DIR = 'CoT_training/models/0-2-model.dat'
MODEL_PATH = "bert-base-uncased" # WORK_DIR + MODEL_DIR

# -----------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
print('Device:', device)

dataset = load_from_disk("datasets/" + DATASET)
print(dataset, dataset['train'][0])

tok: BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
classification_model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
classification_model.classifier = torch.nn.Linear(in_features=768, out_features=MODEL_OUT, bias=True)

classification_trainer = ClassificationTrainer(
    model=classification_model,
    ds=dataset['train'],
    tokenizer=tok,
    identifier_str=IDENTIFIER_STR,
    label_field=LABEL_FIELD
)

classification_trainer.train(4, batch_size=32)

classification_evaluation = Classification_Evaluation(
    model=classification_model,
    ds=dataset,
    tokenizer=tok,
)
print('Train evaluation')
classification_evaluation.evaluate('train')
print('Test evaluation')
classification_evaluation.evaluate('test')