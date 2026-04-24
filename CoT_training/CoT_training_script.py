import torch
import pickle
from datasets import load_from_disk
from transformers import BertTokenizer, BertForMaskedLM

from CoT_Trainer import GeneratorTrainer, EncoderTrainer

VERSION = "CoT_encoder_4eps"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# raw_dataset = load_dataset("kaist-ai/CoT-Collection", trust_remote_code=True, split='train[:1000]')
raw_dataset = load_from_disk("CoT_ds_medium_preprocessed") #"CoT_ds_small"
print("loaded dataset: ", raw_dataset)

name = "bert-base-uncased"     # swap for domain/multilingual BERT as needed
tok: BertTokenizer = BertTokenizer.from_pretrained(name)
model = BertForMaskedLM.from_pretrained('./base_model')

generator_trainer = EncoderTrainer(model, raw_dataset, tok, teacher_forcing_percentage=1, version=VERSION) # GeneratorTrainer
generator_trainer.train(episodes=4, max_generating_steps=64)

generator_trainer.save()