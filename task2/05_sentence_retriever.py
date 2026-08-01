"""
================================================================================
训练 Sentence Retriever (针对passage类型)
================================================================================

目标：训练一个模型，判断passage中的哪个句子包含答案

方法：
1. 用训练数据构造正负样本
   - 正样本：和gold answer最相似的句子
   - 负样本：其他句子
2. 训练二分类器
3. 推理时选择得分最高的句子给T5

================================================================================
"""

# ================================================================================
# 挂载 Google Drive
# ================================================================================
from google.colab import drive
drive.mount('/content/drive')

# ================================================================================
# 安装依赖
# ================================================================================
import subprocess
subprocess.run(['pip', 'install', 'transformers', 'sentence-transformers', 'nltk', 'sentencepiece', '-q'], check=True)

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    T5Tokenizer,
    T5ForConditionalGeneration,
)
from sentence_transformers import SentenceTransformer, util
from google.colab import files
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# ================================================================================
# 配置
# ================================================================================
CONFIG = {
    # Retriever模型
    'retriever_model': 'distilbert-base-uncased',  # 轻量级，稳定
    'retriever_epochs': 3,
    'retriever_batch_size': 16,
    'retriever_lr': 2e-5,

    # T5模型
    't5_model_path': '/content/drive/MyDrive/flan_t5_best_model',

    # 其他
    'max_length': 256,
    'top_k_sentences': 1,  # Oracle实验显示Top-1最好
}

print("=" * 70)
print("Sentence Retriever Training")
print("=" * 70)

# ================================================================================
# 上传数据
# ================================================================================
print("\n请上传 train.jsonl, val.jsonl")
files.upload()

# ================================================================================
# 加载数据
# ================================================================================
def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

train_raw = load_jsonl('/content/train.jsonl')
val_raw = load_jsonl('/content/val.jsonl')

print(f"\n数据: Train={len(train_raw)}, Val={len(val_raw)}")

# ================================================================================
# 辅助函数
# ================================================================================
def get_spoiler_type(item):
    tags = item.get('tags', ['phrase'])
    return tags[0] if isinstance(tags, list) and tags else 'phrase'

def get_spoiler_text(item):
    spoiler = item.get('spoiler', [])
    return ' '.join(spoiler) if isinstance(spoiler, list) else spoiler

def get_sentences(item):
    """把passage切成句子"""
    text = ' '.join(item['targetParagraphs'])
    sentences = sent_tokenize(text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences

# ================================================================================
# 构造Retriever训练数据
# ================================================================================
print("\n" + "=" * 70)
print("构造Retriever训练数据")
print("=" * 70)

# 加载Sentence-BERT用于找正样本
print("\n加载Sentence-BERT...")
sbert = SentenceTransformer('all-MiniLM-L6-v2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sbert.to(device)

def create_retriever_data(raw_data, data_name="data"):
    """
    构造retriever训练数据
    只用passage类型（因为Oracle实验显示passage提升最大）
    """
    retriever_samples = []

    for item in tqdm(raw_data, desc=f"处理{data_name}"):
        # 只处理passage类型
        if get_spoiler_type(item) != 'passage':
            continue

        sentences = get_sentences(item)
        if len(sentences) < 2:
            continue

        reference = get_spoiler_text(item)
        title = item['targetTitle']
        question = ' '.join(item['postText'])
        query = title + ' ' + question

        # 找和reference最相似的句子作为正样本
        ref_emb = sbert.encode(reference, convert_to_tensor=True)
        sent_embs = sbert.encode(sentences, convert_to_tensor=True)
        similarities = util.cos_sim(ref_emb, sent_embs)[0]

        best_idx = torch.argmax(similarities).item()

        # 构造样本
        for idx, sent in enumerate(sentences):
            label = 1 if idx == best_idx else 0
            retriever_samples.append({
                'query': query,
                'sentence': sent,
                'label': label
            })

    return retriever_samples

train_retriever_data = create_retriever_data(train_raw, "训练集")
val_retriever_data = create_retriever_data(val_raw, "验证集")

print(f"\nRetriever训练数据: {len(train_retriever_data)}")
print(f"Retriever验证数据: {len(val_retriever_data)}")

# 统计正负样本比例
train_pos = sum(1 for x in train_retriever_data if x['label'] == 1)
train_neg = len(train_retriever_data) - train_pos
print(f"训练集正负比例: {train_pos}:{train_neg} = 1:{train_neg/train_pos:.1f}")

# ================================================================================
# Retriever Dataset
# ================================================================================
class RetrieverDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 输入格式: [CLS] query [SEP] sentence [SEP]
        encoding = self.tokenizer(
            item['query'],
            item['sentence'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }

# ================================================================================
# 训练Retriever
# ================================================================================
print("\n" + "=" * 70)
print("训练Retriever")
print("=" * 70)

print(f"\n加载模型: {CONFIG['retriever_model']}")
retriever_tokenizer = AutoTokenizer.from_pretrained(CONFIG['retriever_model'])
retriever_model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG['retriever_model'],
    num_labels=2
)

train_dataset = RetrieverDataset(train_retriever_data, retriever_tokenizer, CONFIG['max_length'])
val_dataset = RetrieverDataset(val_retriever_data, retriever_tokenizer, CONFIG['max_length'])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions)
    }

training_args = TrainingArguments(
    output_dir="/content/retriever_model",
    num_train_epochs=CONFIG['retriever_epochs'],
    per_device_train_batch_size=CONFIG['retriever_batch_size'],
    per_device_eval_batch_size=CONFIG['retriever_batch_size'],
    learning_rate=CONFIG['retriever_lr'],
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=100,
    report_to="none",
)

trainer = Trainer(
    model=retriever_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("\n开始训练...")
trainer.train()

# 保存模型
retriever_save_path = '/content/drive/MyDrive/sentence_retriever_model'
retriever_model.save_pretrained(retriever_save_path)
retriever_tokenizer.save_pretrained(retriever_save_path)
print(f"\nRetriever保存到: {retriever_save_path}")

# ================================================================================
# 测试Retriever + T5 Pipeline
# ================================================================================
print("\n" + "=" * 70)
print("测试 Retriever + T5 Pipeline")
print("=" * 70)

# 加载T5
print(f"\n加载T5: {CONFIG['t5_model_path']}")
t5_tokenizer = T5Tokenizer.from_pretrained(CONFIG['t5_model_path'])
t5_model = T5ForConditionalGeneration.from_pretrained(CONFIG['t5_model_path'])
t5_model.to(device)
t5_model.eval()

# Retriever模型准备
retriever_model.to(device)
retriever_model.eval()

def retrieve_top_sentence(item, top_k=1):
    """用训练好的Retriever找最相关的句子"""
    sentences = get_sentences(item)
    if not sentences:
        return ' '.join(item['targetParagraphs'])[:500]

    title = item['targetTitle']
    question = ' '.join(item['postText'])
    query = title + ' ' + question

    # 对每个句子打分
    scores = []
    for sent in sentences:
        encoding = retriever_tokenizer(
            query,
            sent,
            max_length=CONFIG['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = retriever_model(**encoding)
            # 取正类(label=1)的logit
            score = outputs.logits[0, 1].item()
            scores.append(score)

    # 取top-k
    top_indices = np.argsort(scores)[-top_k:][::-1]
    top_sentences = [sentences[i] for i in sorted(top_indices)]

    return ' '.join(top_sentences)

def build_input_with_retrieved(item, retrieved_evidence):
    """用retrieved evidence构建T5输入"""
    spoiler_type = get_spoiler_type(item)
    question = ' '.join(item['postText'])
    context = item['targetTitle'] + ' ' + retrieved_evidence
    return f"type: {spoiler_type} question: {question} context: {context}"

def predict_t5(input_text):
    inputs = t5_tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        return_tensors='pt'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = t5_model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def compute_meteor(predictions, references):
    scores = []
    for pred, ref in zip(predictions, references):
        try:
            pred_tokens = word_tokenize(pred.lower())
            ref_tokens = word_tokenize(ref.lower())
            if ref_tokens and pred_tokens:
                scores.append(meteor_score([ref_tokens], pred_tokens))
        except:
            pass
    return np.mean(scores) if scores else 0.0

# ================================================================================
# 评估：原始 vs Retriever Pipeline
# ================================================================================
print("\n评估验证集...")

# 只评估passage类型
passage_items = [item for item in val_raw if get_spoiler_type(item) == 'passage']
print(f"Passage类型样本数: {len(passage_items)}")

# 原始方法
preds_original = []
refs = []

for item in tqdm(passage_items, desc="原始方法"):
    spoiler_type = get_spoiler_type(item)
    question = ' '.join(item['postText'])
    context = item['targetTitle'] + ' ' + ' '.join(item['targetParagraphs'])
    input_text = f"type: {spoiler_type} question: {question} context: {context}"

    pred = predict_t5(input_text)
    ref = get_spoiler_text(item)

    preds_original.append(pred)
    refs.append(ref)

meteor_original = compute_meteor(preds_original, refs)
print(f"\n原始方法 (passage) METEOR: {meteor_original:.4f}")

# Retriever方法
preds_retriever = []

for item in tqdm(passage_items, desc="Retriever方法"):
    retrieved = retrieve_top_sentence(item, top_k=1)
    input_text = build_input_with_retrieved(item, retrieved)
    pred = predict_t5(input_text)
    preds_retriever.append(pred)

meteor_retriever = compute_meteor(preds_retriever, refs)
print(f"Retriever方法 (passage) METEOR: {meteor_retriever:.4f} ({'+' if meteor_retriever > meteor_original else ''}{meteor_retriever - meteor_original:.4f})")

# ================================================================================
# 全类型评估
# ================================================================================
print("\n" + "=" * 70)
print("全类型评估")
print("=" * 70)

preds_all_original = []
preds_all_retriever = []
refs_all = []
types_all = []

for item in tqdm(val_raw, desc="全类型评估"):
    spoiler_type = get_spoiler_type(item)
    ref = get_spoiler_text(item)

    # 原始方法
    question = ' '.join(item['postText'])
    context = item['targetTitle'] + ' ' + ' '.join(item['targetParagraphs'])
    input_original = f"type: {spoiler_type} question: {question} context: {context}"
    pred_original = predict_t5(input_original)

    # Retriever方法（只对passage使用）
    if spoiler_type == 'passage':
        retrieved = retrieve_top_sentence(item, top_k=1)
        input_retriever = build_input_with_retrieved(item, retrieved)
        pred_retriever = predict_t5(input_retriever)
    else:
        pred_retriever = pred_original  # 其他类型保持原样

    preds_all_original.append(pred_original)
    preds_all_retriever.append(pred_retriever)
    refs_all.append(ref)
    types_all.append(spoiler_type)

meteor_all_original = compute_meteor(preds_all_original, refs_all)
meteor_all_retriever = compute_meteor(preds_all_retriever, refs_all)

print(f"\n全类型结果:")
print(f"  原始方法:     {meteor_all_original:.4f}")
print(f"  Retriever:    {meteor_all_retriever:.4f} ({'+' if meteor_all_retriever > meteor_all_original else ''}{meteor_all_retriever - meteor_all_original:.4f})")

# 分类型
print("\n分类型结果:")
for t in ['phrase', 'passage', 'multi']:
    indices = [i for i, x in enumerate(types_all) if x == t]
    if not indices:
        continue

    t_orig = [preds_all_original[i] for i in indices]
    t_retr = [preds_all_retriever[i] for i in indices]
    t_ref = [refs_all[i] for i in indices]

    m_orig = compute_meteor(t_orig, t_ref)
    m_retr = compute_meteor(t_retr, t_ref)

    print(f"  {t}: {m_orig:.4f} → {m_retr:.4f} ({'+' if m_retr > m_orig else ''}{m_retr - m_orig:.4f})")

# ================================================================================
# 最终结果
# ================================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)
print(f"\n原始方法:  {meteor_all_original:.4f}")
print(f"Retriever: {meteor_all_retriever:.4f}")
print(f"提升:      {'+' if meteor_all_retriever > meteor_all_original else ''}{meteor_all_retriever - meteor_all_original:.4f}")
print("=" * 70)
