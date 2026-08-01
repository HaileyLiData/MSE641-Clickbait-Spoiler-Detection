# Task 1 - Spoiler Type Classification using DeBERTa-v3-large
# Final attempt: MAX_LEN=512, no class weights, run on a T4 (16GB) GPU
#
# Requirements (run once in terminal):
#   pip install torch torchvision torchaudio
#   pip install transformers==4.40.0 sentencepiece scikit-learn accelerate
#
# Usage:
#   python task1-deberta-v3-large-final.py
#
# Training time: expect longer than the base model, roughly 1-1.5 hours on a T4

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score

LABEL2ID = {'phrase': 0, 'passage': 1, 'multi': 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
MODEL_NAME = 'microsoft/deberta-v3-large'
MAX_LEN = 512
TRAIN_FILE = 'train.jsonl'   # put train.jsonl in the same folder as this script
VAL_FILE = 'val.jsonl'       # put val.jsonl in the same folder as this script
OUTPUT_DIR = './task1-deberta-v3-large-output'


def build_text(row):
    post = ' '.join(row['postText'])
    title = row['targetTitle']
    paragraphs = ' '.join(row['targetParagraphs'])
    return post + ' - ' + title + ' ' + paragraphs


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
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer(
            item['text'],
            max_length=MAX_LEN,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels': torch.tensor(item['label'], dtype=torch.long),
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {'accuracy': accuracy_score(labels, preds)}


def main():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Using device: {device}')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True,
    )

    train_data = load_data(TRAIN_FILE)
    val_data = load_data(VAL_FILE)
    print(f'Train: {len(train_data)}, Val: {len(val_data)}')

    train_dataset = SpoilerDataset(train_data, tokenizer)
    val_dataset = SpoilerDataset(val_data, tokenizer)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        warmup_steps=200,
        weight_decay=0.01,
        max_grad_norm=1.0,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        logging_steps=50,
        fp16=(device == 'cuda'),
        gradient_checkpointing=True,
        report_to='none',
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    results = trainer.evaluate()
    print(f'\nFinal Val accuracy: {results["eval_accuracy"]:.4f}')

    preds_output = trainer.predict(val_dataset)
    pred_labels = np.argmax(preds_output.predictions, axis=1)
    with open('task1-deberta-v3-large-val-preds.jsonl', 'w') as f:
        for item, pred in zip(val_data, pred_labels):
            f.write(json.dumps({'uuid': item['uuid'], 'spoilerType': ID2LABEL[pred]}) + '\n')
    print('Predictions saved to task1-deberta-v3-large-val-preds.jsonl')


if __name__ == '__main__':
    main()
