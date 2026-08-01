import subprocess
subprocess.run(['pip', 'install', 'transformers', 'sentencepiece', 'scikit-learn', 'accelerate', '-q'], check=True)

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
from google.colab import files

LABEL2ID = {'phrase': 0, 'passage': 1, 'multi': 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def build_text(row):
    return ' '.join(row['postText']) + ' - ' + row['targetTitle'] + ' ' + ' '.join(row['targetParagraphs'])

def load_data(path):
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            rows.append({
                'text': build_text(d),
                'label': LABEL2ID[d['tags'][0]],
                'uuid': d.get('uuid', d.get('id', '')),
            })
    return rows

class SpoilerDataset(Dataset):
    def __init__(self, data, tok):
        self.data = data
        self.tok = tok
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        enc = self.tok(self.data[i]['text'], max_length=512, truncation=True, padding='max_length', return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels': torch.tensor(self.data[i]['label'], dtype=torch.long),
        }

def compute_metrics(p):
    return {'accuracy': accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))}

print("请上传 train.jsonl 和 val.jsonl")
files.upload()

tok = AutoTokenizer.from_pretrained('microsoft/deberta-base')
model = AutoModelForSequenceClassification.from_pretrained(
    'microsoft/deberta-base', num_labels=3, label2id=LABEL2ID, id2label=ID2LABEL)

train_data = load_data('/content/train.jsonl')
val_data = load_data('/content/val.jsonl')
print(f'Train: {len(train_data)}, Val: {len(val_data)}')

args = TrainingArguments(
    output_dir='/content/output',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=200,
    weight_decay=0.01,
    max_grad_norm=1.0,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    logging_steps=50,
    fp16=False,
    report_to='none',
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=SpoilerDataset(train_data, tok),
    eval_dataset=SpoilerDataset(val_data, tok),
    compute_metrics=compute_metrics,
)

trainer.train()

results = trainer.evaluate()
print(f'\nFinal Val accuracy: {results["eval_accuracy"]:.4f}')

pred = trainer.predict(SpoilerDataset(val_data, tok))
pred_labels = np.argmax(pred.predictions, axis=1)
with open('/content/task1-roberta-preds.jsonl', 'w') as f:
    for item, p in zip(val_data, pred_labels):
        f.write(json.dumps({'uuid': item['uuid'], 'spoilerType': ID2LABEL[p]}) + '\n')
print('Done. Download task1-roberta-preds.jsonl from Files panel.')
