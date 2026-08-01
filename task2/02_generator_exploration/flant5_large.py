# Flan-T5-large instead of Flan-T5-base (Attempt 13) / vanilla T5-base (Attempt 3).
# Same recipe as the Flan-T5-base script (task2-flant5-colab.py) -- only the
# checkpoint changes, plus a smaller per-step batch (large model needs more
# memory per sequence than base).
#
# Motivation: TohokuNLP's SemEval-2023 Task 5 system paper (same 3200-example
# Webis Clickbait Spoiling Corpus we're using) found flan-t5-large clearly beats
# both t5-large and flan-t5-base on validation METEOR (0.5124 vs 0.4872 vs
# 0.3685) -- the Flan advantage only shows up at large scale, consistent with
# our own Attempt 13 finding that Flan-T5-base did NOT beat vanilla T5-base.

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
DRIVE_DIR = '/content/drive/MyDrive/MSE641_Task2_FlanT5Large'
import os
os.makedirs(DRIVE_DIR, exist_ok=True)

MODEL_NAME = 'google/flan-t5-large'
MAX_INPUT_LEN = 512
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
        # no padding here -- DataCollatorForSeq2Seq below pads dynamically to the
        # longest sequence in each batch, instead of wasting compute/memory padding
        # every example to the full 512/128 regardless of its real length
        enc = self.tok(
            item['input'],
            max_length=MAX_INPUT_LEN,
            truncation=True,
        )
        result = {
            'input_ids': enc['input_ids'],
            'attention_mask': enc['attention_mask'],
        }
        if not self.is_test:
            target = self.tok(
                text_target=item['target'],
                max_length=MAX_TARGET_LEN,
                truncation=True,
            )
            result['labels'] = target['input_ids']
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
model.config.use_cache = False  # incompatible with gradient checkpointing during training; avoids warnings/wasted memory

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
print('Use BF16:', use_bf16)

train_data = load_data(TRAIN_FILE)
val_data = load_data(VAL_FILE)
print(f'Train: {len(train_data)}, Val: {len(val_data)}')

train_dataset = SpoilerDataset(train_data, tok)
val_dataset = SpoilerDataset(val_data, tok)

args = Seq2SeqTrainingArguments(
    output_dir='/content/task2-flant5-large-output',
    num_train_epochs=5,               # reduced from Attempt 13's 10 -- large model, keep wall-clock bounded.
                                       # load_best_model_at_end only protects against OVERFITTING (picks the
                                       # best-so-far epoch instead of the last one) -- it does NOT protect
                                       # against undertraining, so 5 is a real time/thoroughness tradeoff, not
                                       # a free safety net
    per_device_train_batch_size=1,    # large model needs much more memory per sequence than base (which used 2)
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,   # effective batch 16, same as Attempt 13's 2*8
    learning_rate=5e-5,               # lower than Attempt 13's 1e-4 -- full fine-tuning of a large model is
                                       # less stable at the base-size learning rate
    warmup_ratio=0.1,                 # ~1000 total optimizer steps here (3200/16*5) -- a ratio scales
                                       # correctly regardless of exact data/batch size, unlike a fixed step count
    weight_decay=0.01,
    optim='adafactor',                # AdamW's optimizer state (2 extra moments per param, fp32) would need
                                       # ~12GB just for a 770M model's weights+grads+optimizer state, before any
                                       # activations -- Adafactor is the standard memory-saving choice for
                                       # T5-scale training and is what the original T5 paper itself used
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=2,               # large-model checkpoints are big; cap how many pile up on Colab's disk
    load_best_model_at_end=True,
    metric_for_best_model='meteor',
    predict_with_generate=True,
    generation_max_length=128,
    logging_steps=50,
    seed=42,
    data_seed=42,
    fp16=False,                      # T5/Flan-T5 has a known NaN overflow issue specifically under fp16 (Attempt 13) --
                                      # bf16's wider exponent range avoids that failure mode, so use it when the GPU supports it
    bf16=use_bf16,
    gradient_checkpointing=True,     # large model (770M) on a 16GB T4 -- trade compute for memory preemptively,
                                      # since batch is already at the minimum (1) with no room to shrink further if it OOMs
    report_to='none',
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model, padding=True, label_pad_token_id=-100)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model(f'{DRIVE_DIR}/model')
tok.save_pretrained(f'{DRIVE_DIR}/model')
print(f'Model saved to {DRIVE_DIR}/model')

results = trainer.evaluate()
print(f'\nMETEOR: {results["eval_meteor"]:.4f}')
print('(compare to: Attempt 3 t5-base 0.4257, Attempt 13 flan-t5-base 0.4234, TohokuNLP paper flan-t5-large val 0.5124)')

val_preds = trainer.predict(val_dataset)
val_predictions = np.array(val_preds.predictions)
val_predictions = np.where(
    (val_predictions < 0) | (val_predictions >= tok.vocab_size),
    tok.pad_token_id,
    val_predictions,
).astype(np.int64)
decoded = tok.batch_decode(val_predictions.tolist(), skip_special_tokens=True)
with open(f'{DRIVE_DIR}/task2-flant5-large-val-preds.jsonl', 'w') as f:
    for item, pred in zip(val_data, decoded):
        f.write(json.dumps({'uuid': item['id'], 'spoiler': pred}) + '\n')
print(f'Val predictions saved to {DRIVE_DIR}/task2-flant5-large-val-preds.jsonl')
