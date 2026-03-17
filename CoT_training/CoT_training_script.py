from datasets import load_from_disk
import pickle
from transformers import BertTokenizer, BertForMaskedLM

from CoT_Trainer import GeneratorTrainer

version = 1

# raw_dataset = load_dataset("kaist-ai/CoT-Collection", trust_remote_code=True, split='train[:1000]')
raw_dataset = load_from_disk("CoT_ds_small")
print("loaded dataset: ", raw_dataset)

name = "bert-base-uncased"     # swap for domain/multilingual BERT as needed
tok: BertTokenizer = BertTokenizer.from_pretrained(name)
model = BertForMaskedLM.from_pretrained('./base_model')

generator_trainer = GeneratorTrainer(model, raw_dataset, tok)
generator_trainer.train(steps=0)

with open(f'models/{version}-losses.dat', 'wb') as f:
    pickle.dump(generator_trainer.losses, f)

generator_trainer.model.save_pretrained(f'models/{version}-model.dat')