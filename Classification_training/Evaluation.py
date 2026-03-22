# pip install -q transformers datasets accelerate torch==2.* sentencepiece
import torch
from dataclasses import dataclass
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class Classification_Evaluation:
    model: BertForSequenceClassification
    ds: Dataset
    tokenizer: BertTokenizer
    text_field: str = 'text'
    label_field: str = 'label'

    def evaluate(self, split):
        self.model.eval()
        ds_split: Dataset = self.ds[split]
        predictions = []
        for batch in ds_split.batch(batch_size=16):
            tokenized_text = self.tokenizer(list(batch[self.text_field]), truncation=True, return_tensors='pt', padding=True).to(device)
            output = self.model.forward(**tokenized_text)
            prediction = torch.argmax(output.logits, dim=1)
            predictions.append(prediction)
        all_predictions = torch.cat(predictions).cpu().numpy()
        print(classification_report(y_true=ds_split[self.label_field], y_pred=all_predictions))
        return classification_report(y_true=ds_split[self.label_field], y_pred=all_predictions, output_dict=True)