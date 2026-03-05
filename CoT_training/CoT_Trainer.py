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


    def tokenize(self, features: List[Dict]) -> tuple[torch.Tensor, torch.Tensor]:
        questions = self.tokenizer(list(features["source"]), add_special_tokens=False, return_tensors='pt', padding=True).to(device)
        answers = self.tokenizer(list(features["rationale"]), add_special_tokens=False, return_tensors='pt', padding=True).to(device)
        return questions['input_ids'], answers['input_ids']


    def generate(self, questions: torch.Tensor, answers: torch.Tensor|None=None, max_length=200):
        generated_answers = []
        logits = []
        batch_size = questions.shape[0]
        cls_token_tensor = self.tokenizer.cls_token_id * torch.ones((batch_size, 1)).to(device)
        mask_token_tensor = self.tokenizer.mask_token_id * torch.ones((batch_size, 1)).to(device)
        sep_token_tensor = self.tokenizer.sep_token_id * torch.ones((batch_size, 1)).to(device)
        for i in range(answers.shape[1] if answers is not None else max_length):
            teacher_forcing_enabled = (answers is not None) and (i < len(answers))
            use_teacher_forcing = (answers is not None) and (i < len(answers)) and (random.random() < self.teacher_forcing_percentage)
            prefix = answers[:, :i] if use_teacher_forcing else (torch.stack(generated_answers, axis=1) if len(generated_answers)>0 else torch.zeros((batch_size, 0)))

            #print(*(x.size() for x in (cls_token_tensor, questions, prefix, mask_token_tensor, sep_token_tensor)))
            inp = torch.concat((cls_token_tensor, questions, prefix, mask_token_tensor, sep_token_tensor), dim=1)
            mask_pos = -2
            attention_mask = torch.ones(inp.shape).to(device)
            attention_mask[:,mask_pos] = 0
            token_type_ids = torch.concat((torch.zeros((batch_size, questions.shape[1]+1)), torch.ones((batch_size, prefix.shape[1]+2)).to(device)), dim=1)

            #print(*(x.size() for x in (inp.int(), token_type_ids.int(), attention_mask.int())))
            generated = self.model.forward(input_ids=inp.int(), token_type_ids=token_type_ids.int(), attention_mask=attention_mask.int())

            generated_answers.append(torch.argmax(generated.logits[:,mask_pos], dim=-1))
            logits.append(generated.logits[:,mask_pos])
            torch.cuda.empty_cache()
        #print(generated_answers, logits)
        print(torch.stack(generated_answers).shape)
        print(len(generated_answers))
        #print(torch.stack(generated_answers).size(), torch.stack(logits).size())
        return torch.stack(generated_answers, dim=1), torch.stack(logits, dim=1)

    def generate_step(self, i: int, questions: torch.Tensor, answers: torch.Tensor, generated_answers: torch.Tensor, batch_size: int, cls_token_tensor:torch.Tensor, mask_token_tensor:torch.Tensor, sep_token_tensor:torch.Tensor):
        use_teacher_forcing = (answers is not None) and (i < len(answers)) and (random.random() < self.teacher_forcing_percentage)
        prefix = answers[:, :i] if use_teacher_forcing else (torch.stack(generated_answers, axis=1) if len(generated_answers)>0 else torch.zeros((batch_size, 0)))

        #print(*(x.size() for x in (cls_token_tensor, questions, prefix, mask_token_tensor, sep_token_tensor)))
        inp = torch.concat((cls_token_tensor, questions, prefix, mask_token_tensor, sep_token_tensor), dim=1)
        mask_pos = -2
        attention_mask = torch.ones(inp.shape).to(device)
        attention_mask[:,mask_pos] = 0
        token_type_ids = torch.concat((torch.zeros((batch_size, questions.shape[1]+1)), torch.ones((batch_size, prefix.shape[1]+2)).to(device)), dim=1)

        #print(*(x.size() for x in (inp.int(), token_type_ids.int(), attention_mask.int())))
        generated = self.model.forward(input_ids=inp.int(), token_type_ids=token_type_ids.int(), attention_mask=attention_mask.int())
        return torch.argmax(generated.logits[:,mask_pos], dim=-1), generated.logits[:,mask_pos]

    def train(self, steps, batch_size=16, max_generate_length=200, max_generating_steps=64):
        cls_token_tensor = self.tokenizer.cls_token_id * torch.ones((batch_size, 1)).to(device)
        mask_token_tensor = self.tokenizer.mask_token_id * torch.ones((batch_size, 1)).to(device)
        sep_token_tensor = self.tokenizer.sep_token_id * torch.ones((batch_size, 1)).to(device)
        special_tokens = (cls_token_tensor, mask_token_tensor, sep_token_tensor)

        for step in range(steps):
            i_samples = np.random.randint(0, len(self.ds), batch_size)
            samples = self.ds.select(i_samples)
            print(samples)
            answers, questions = self.tokenize(samples)

            generated_answers = []
            logits = []
            for i in range(answers.shape[1] if answers is not None else max_generate_length):
                generated_answer, step_logits = self.generate_step(i, questions, answers, generated_answers, batch_size, *special_tokens)
                generated_answers.append(generated_answer)
                logits.append(step_logits)

                if (i+1) % max_generating_steps == 0:
                    loss = self.loss_func(torch.stack(logits, dim=-1), answers[:, i-max_generating_steps+1: i+1])
                    loss.backward()
                    self.optimizer.step()
                    logits = []
                    print(f'{step} - loss: {loss.item()}')
                    #TODO: fix last step