"""
Task 1 - DeBERTa V1-base 组合优化版
整合所有改进方法:
1. FGM对抗训练 (+1-2%)
2. Class Weights类别平衡 (+0.5-1%)
3. Cosine Scheduler + 更多Epochs
4. Early Stopping防止过拟合

目标: 从0.765 → 0.78+
"""

import subprocess
subprocess.run(['pip', 'install', 'transformers', 'sentencepiece', 'scikit-learn', 'accelerate', '-q'], check=True)

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from google.colab import files

# ============== 可调参数 ==============
CONFIG = {
    'model_name': 'microsoft/deberta-base',
    'max_length': 512,
    'learning_rate': 2e-5,        # 已验证最佳
    'num_train_epochs': 5,        # 固定5，8会过拟合
    'per_device_train_batch_size': 16,  # FGM需要额外显存，用16
    'warmup_ratio': 0.1,
    'lr_scheduler_type': 'cosine',
    'weight_decay': 0.01,
    'fgm_epsilon': 0.5,           # FGM扰动强度
    'early_stopping_patience': 2,
}

# ============== FGM 对抗训练 ==============
class FGM:
    def __init__(self, model, epsilon=0.5, emb_name='word_embeddings'):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# ============== 组合Trainer: FGM + Class Weights ==============
class CombinedTrainer(Trainer):
    """FGM对抗训练 + 类别权重"""

    def __init__(self, fgm_epsilon=0.5, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fgm = None
        self.fgm_epsilon = fgm_epsilon
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = torch.tensor(self.class_weights, dtype=torch.float).to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)

        # 把labels放回去，后续可能需要
        inputs["labels"] = labels

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.fgm is None:
            self.fgm = FGM(model, epsilon=self.fgm_epsilon)

        model.train()
        inputs = self._prepare_inputs(inputs)

        # 正常前向传播
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        # 反向传播
        self.accelerator.backward(loss)

        # FGM对抗
        self.fgm.attack()
        with self.compute_loss_context_manager():
            loss_adv = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss_adv = loss_adv.mean()
        self.accelerator.backward(loss_adv)
        self.fgm.restore()

        return loss + loss_adv

# ============== 数据处理 ==============
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
    def __init__(self, data, tok, max_length):
        self.data = data
        self.tok = tok
        self.max_length = max_length
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        enc = self.tok(self.data[i]['text'], max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels': torch.tensor(self.data[i]['label'], dtype=torch.long),
        }

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_score(p.label_ids, preds)
    f1 = f1_score(p.label_ids, preds, average='macro')
    return {'accuracy': acc, 'f1_macro': f1}

# ============== 主流程 ==============
print("=" * 60)
print("Task 1 - DeBERTa V1-base Combined Optimization")
print("FGM + Class Weights + Cosine Scheduler + Early Stopping")
print("=" * 60)
print("\n配置:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")
print("=" * 60)

print("\n请上传 train.jsonl 和 val.jsonl")
files.upload()

# 加载数据
train_data = load_data('/content/train.jsonl')
val_data = load_data('/content/val.jsonl')
print(f'Train: {len(train_data)}, Val: {len(val_data)}')

# 计算类别权重
train_labels = np.array([d['label'] for d in train_data])
class_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=train_labels)
print(f'\nClass weights: phrase={class_weights[0]:.3f}, passage={class_weights[1]:.3f}, multi={class_weights[2]:.3f}')

# 模型和tokenizer
tok = AutoTokenizer.from_pretrained(CONFIG['model_name'])
model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG['model_name'], num_labels=3, label2id=LABEL2ID, id2label=ID2LABEL)

# 训练参数
args = TrainingArguments(
    output_dir='/content/output_combined',
    num_train_epochs=CONFIG['num_train_epochs'],
    per_device_train_batch_size=CONFIG['per_device_train_batch_size'],
    per_device_eval_batch_size=16,
    learning_rate=CONFIG['learning_rate'],
    warmup_ratio=CONFIG['warmup_ratio'],
    lr_scheduler_type=CONFIG['lr_scheduler_type'],
    weight_decay=CONFIG['weight_decay'],
    max_grad_norm=1.0,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',
    greater_is_better=True,
    logging_steps=50,
    fp16=False,
    report_to='none',
)

# Combined Trainer
trainer = CombinedTrainer(
    fgm_epsilon=CONFIG['fgm_epsilon'],
    class_weights=class_weights.tolist(),
    model=model,
    args=args,
    train_dataset=SpoilerDataset(train_data, tok, CONFIG['max_length']),
    eval_dataset=SpoilerDataset(val_data, tok, CONFIG['max_length']),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=CONFIG['early_stopping_patience'])],
)

print("\n开始训练 (Combined: FGM + Class Weights + Cosine)...")
trainer.train()

# 评估
results = trainer.evaluate()
print(f'\n{"="*60}')
print(f'Final Results:')
print(f'  Accuracy: {results["eval_accuracy"]:.4f}')
print(f'  F1 Macro: {results["eval_f1_macro"]:.4f}')
print(f'{"="*60}')

# 详细分类报告
pred = trainer.predict(SpoilerDataset(val_data, tok, CONFIG['max_length']))
pred_labels = np.argmax(pred.predictions, axis=1)
true_labels = [d['label'] for d in val_data]
print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=['phrase', 'passage', 'multi']))

# 保存预测结果
with open('/content/task1-v1-combined-preds.jsonl', 'w') as f:
    for item, p in zip(val_data, pred_labels):
        f.write(json.dumps({'uuid': item['uuid'], 'spoilerType': ID2LABEL[p]}) + '\n')

print('\nDone! Download task1-v1-combined-preds.jsonl from Files panel.')
print('\n如果效果好，可以生成test.jsonl的预测用于Kaggle提交')
