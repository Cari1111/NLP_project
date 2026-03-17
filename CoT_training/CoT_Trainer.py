# pip install -q transformers datasets accelerate torch==2.* sentencepiece
import random, torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from datasets import load_dataset, Dataset
from transformers import BertTokenizer, BertForMaskedLM, BertForSequenceClassification
from torch.utils.data import default_collate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
print('Device:', device)

@dataclass
class GeneratorTrainer:
    model: BertForMaskedLM
    ds: Dataset
    tokenizer: BertTokenizer
    teacher_forcing_percentage: float = 0.8

    def __post_init__(self):
        self.model.to(device)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.cls_token_tensor = torch.tensor([[self.tokenizer.cls_token_id]])
        self.losses = []

    def tokenize(self, features: List[Dict]) -> tuple[torch.Tensor, torch.Tensor]:
        questions = self.tokenizer(features["source"], add_special_tokens=False, return_tensors='pt', padding=True).to(device)
        answers = self.tokenizer(features["rationale"], add_special_tokens=False, return_tensors='pt', padding=True).to(device)
        return questions['input_ids'][0], answers['input_ids'][0]

    def generate_step(self, i: int, questions: torch.Tensor, answers: torch.Tensor, generated_answers: torch.Tensor, cls_token_tensor:torch.Tensor, mask_token_tensor:torch.Tensor, sep_token_tensor:torch.Tensor):
        use_teacher_forcing = (answers is not None) and (i < len(answers)) and (random.random() < self.teacher_forcing_percentage)
        prefix = answers[:i] if use_teacher_forcing else (torch.stack(generated_answers) if len(generated_answers)>0 else torch.tensor([]))

        #print(*(x.size() for x in (cls_token_tensor, questions, prefix, mask_token_tensor, sep_token_tensor)))
        inp = torch.concat((cls_token_tensor, questions, prefix, mask_token_tensor, sep_token_tensor))
        mask_pos = -2
        attention_mask = torch.ones(inp.shape).to(device)
        attention_mask[mask_pos] = 0
        token_type_ids = torch.concat((torch.zeros((questions.shape[0]+1)), torch.ones((prefix.shape[0]+2)).to(device)))

        #print(*(x.size() for x in (inp.int(), token_type_ids.int(), attention_mask.int())))
        generated = self.model.forward(input_ids=inp.unsqueeze(0).int(), token_type_ids=token_type_ids.unsqueeze(0).int(), attention_mask=attention_mask.unsqueeze(0).int())
        logits = generated.logits.squeeze()[mask_pos]
        #print(*(x.size() for x in (generated.logits, logits)))
        return torch.argmax(logits), logits

    def train(self, steps, max_generate_length=200, max_generating_steps=64):
        cls_token_tensor =  torch.tensor([self.tokenizer.cls_token_id]).to(device)
        mask_token_tensor = torch.tensor([self.tokenizer.mask_token_id]).to(device)
        sep_token_tensor = torch.tensor([self.tokenizer.sep_token_id]).to(device)
        special_tokens = (cls_token_tensor, mask_token_tensor, sep_token_tensor)

        for step in range(steps):
            sample = self.ds[np.random.randint(0, len(self.ds))]
            #print(sample)
            questions, answers = self.tokenize(sample)
            if questions.size()[0] < answers.size()[0]: print(f'question: {questions.size()[0]}, answer {answers.size()[0]}', sample)

            generated_answers = []
            logits = []
            max_iter = answers.shape[0] if answers is not None else max_generate_length
            for i in range(max_iter):
                generated_answer, step_logits = self.generate_step(i, questions, answers, generated_answers, *special_tokens)
                generated_answers.append(generated_answer)
                logits.append(step_logits)

                if (i+1) % max_generating_steps == 0 or i+1 == max_iter:
                    iter_training_cycle = (i % max_generating_steps) + 1
                    #print(*(x.size() for x in (torch.stack(logits, dim=-1), answers[i-iter_training_cycle+1: i+1])))
                    loss = self.loss_func(torch.stack(logits), answers[i-iter_training_cycle+1: i+1])
                    loss.backward()
                    self.optimizer.step()
                    logits = []
                    print(f'{step} - loss: {loss.item()}')
                    self.losses.append(loss.item())