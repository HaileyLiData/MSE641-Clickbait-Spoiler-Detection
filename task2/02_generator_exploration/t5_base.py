import subprocess
subprocess.run(['pip', 'install', 'transformers', 'sentencepiece', 'scikit-learn', 'accelerate', 'nltk', '-q'], check=True)

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.translate.meteor_score import meteor_score
from google.colab import files, drive

drive.mount('/content/drive')
DRIVE_DIR = '/content/drive/MyDrive/MSE641_Task2'
import os
os.makedirs(DRIVE_DIR, exist_ok=True)

MODEL_NAME = 't5-base'
MAX_INPUT_LEN = 1024
MAX_TARGET_LEN = 128
TRAIN_FILE = '/content/train.jsonl'
VAL_FILE = '/content/val.jsonl'


def build_input(row):
    post = ' '.join(row['postText'])
    title = row['targetTitle']
    paragraphs = ' '.join(row['targetParagraphs'])
    return 'question: ' + post + ' context: ' + title + ' ' + paragraphs


def load_data(path, is_test=False):
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            row = {
                'input': build_input(d),
                'id': d.get('uuid', d.get('id', '')),
            }
            if not is_test:
                row['target'] = ' '.join(d['spoiler'])
            rows.append(row)
    return rows


class SpoilerDataset(Dataset):
    def __init__(self, data, tokenizer, is_test=False):
        self.data = data
        self.tok = tokenizer
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        enc = self.tok(
            item['input'],
            max_length=MAX_INPUT_LEN,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        result = {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
        }
        if not self.is_test:
            target = self.tok(
                text_target=item['target'],
                max_length=MAX_TARGET_LEN,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
            )
            labels = target['input_ids'].squeeze()
            labels[labels == self.tok.pad_token_id] = -100
            result['labels'] = labels
        return result


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.array(predictions)
    predictions = np.where(
        (predictions < 0) | (predictions >= tok.vocab_size),
        tok.pad_token_id,
        predictions,
    ).astype(np.int64)
    decoded_preds = tok.batch_decode(predictions.tolist(), skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tok.pad_token_id).astype(np.int64)
    decoded_labels = tok.batch_decode(labels.tolist(), skip_special_tokens=True)

    scores = []
    for pred, label in zip(decoded_preds, decoded_labels):
        score = meteor_score([label.split()], pred.split())
        scores.append(score)

    return {'meteor': np.mean(scores)}


print("请上传 train.jsonl 和 val.jsonl")
files.upload()

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

train_data = load_data(TRAIN_FILE)
val_data = load_data(VAL_FILE)
print(f'Train: {len(train_data)}, Val: {len(val_data)}')

train_dataset = SpoilerDataset(train_data, tok)
val_dataset = SpoilerDataset(val_data, tok)

args = Seq2SeqTrainingArguments(
    output_dir='/content/task2-t5-output',
    num_train_epochs=10,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    warmup_steps=200,
    weight_decay=0.01,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='meteor',
    predict_with_generate=True,
    generation_max_length=128,
    logging_steps=50,
    fp16=True,
    report_to='none',
)

data_collator = DataCollatorForSeq2Seq(tok, model=model, padding=True)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

trainer.train()

# save model to Drive immediately, before anything else can go wrong
trainer.save_model(f'{DRIVE_DIR}/model')
tok.save_pretrained(f'{DRIVE_DIR}/model')
print(f'Model saved to {DRIVE_DIR}/model')

results = trainer.evaluate()
print(f'\nMETEOR: {results["eval_meteor"]:.4f}')

# save val predictions directly to Drive
val_preds = trainer.predict(val_dataset)
val_predictions = np.array(val_preds.predictions)
val_predictions = np.where(
    (val_predictions < 0) | (val_predictions >= tok.vocab_size),
    tok.pad_token_id,
    val_predictions,
).astype(np.int64)
decoded = tok.batch_decode(val_predictions.tolist(), skip_special_tokens=True)
with open(f'{DRIVE_DIR}/task2-t5-val-preds.jsonl', 'w') as f:
    for item, pred in zip(val_data, decoded):
        f.write(json.dumps({'uuid': item['id'], 'spoiler': pred}) + '\n')
print(f'Val predictions saved to {DRIVE_DIR}/task2-t5-val-preds.jsonl')
