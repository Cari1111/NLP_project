import datasets
from datasets import load_dataset, load_from_disk
from transformers import BertTokenizer, BertForMaskedLM

from CoT_Trainer import GeneratorTrainer

# raw_dataset = load_dataset("kaist-ai/CoT-Collection", trust_remote_code=True, split='train[:1000]')
raw_dataset = load_from_disk("CoT_ds_small")
print("loaded dataset: ", raw_dataset)

ckpt = "bert-base-uncased"     # swap for domain/multilingual BERT as needed
tok: BertTokenizer = BertTokenizer.from_pretrained(ckpt)
model = BertForMaskedLM.from_pretrained(ckpt)

generator_trainer = GeneratorTrainer(model, raw_dataset, tok)
generator_trainer.train(steps=10, batch_size=2)