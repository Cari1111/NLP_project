from datasets import load_dataset
from transformers import BertTokenizer, BertForMaskedLM

from CoT_Trainer import GeneratorTrainer

raw_dataset = load_dataset("kaist-ai/CoT-Collection", trust_remote_code=True, split='train[:1000]')
print("loaded dataset: ", raw_dataset)

ckpt = "bert-base-uncased"     # swap for domain/multilingual BERT as needed
tok: BertTokenizer = BertTokenizer.from_pretrained(ckpt)
model = BertForMaskedLM.from_pretrained(ckpt)

generator_trainer = GeneratorTrainer(model, raw_dataset, tok)
generator_trainer.train(episodes=10, batch_size=2)