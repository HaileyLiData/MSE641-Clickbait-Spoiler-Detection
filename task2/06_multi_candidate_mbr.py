#!/usr/bin/env python3
"""
最终提交（0.45412）的正式脚本 - 之前是在命令行里临时跑的，现在补成正式脚本方便复现。

做的事：读取 task2-mbr-test-candidate-pool.jsonl（GPU那边已经生成好的、每条
passage样本23个原始候选），只取每条的前13个（对齐 Hailey 原始配方的候选数量），
去重后用 token-F1 consensus（"少数服从多数"）选一个作为最终答案；
phrase/multi 直接用候选池里存的那个单一预测（跟 Hailey 原来的做法一致，没有改动）。

跟 submission_task2_tokenf1_budget13.csv 应该是完全一样的结果（同一份候选池，
同一套选择逻辑），这个脚本只是把当时的临时代码正式留档。

用法：
    python task2-mbr-final-submission-budget13.py
"""
import json
import csv
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize

for pkg, sub in [('punkt', 'tokenizers'), ('punkt_tab', 'tokenizers')]:
    try:
        nltk.data.find(f'{sub}/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)

CANDIDATE_POOL_PATH = 'task2-mbr-test-candidate-pool.jsonl'
OUTPUT_PATH = 'submission_task2_final_0.45412.csv'
BUDGET = 13  # matches Hailey's original recipe size (beam2/4/6/8 + 3 temps x2 + 1 diverse-beam call x3)


def normalize_candidate(text):
    return ' '.join(text.lower().split())


def dedup_texts(raw_items):
    seen, out = set(), []
    for item in raw_items:
        key = normalize_candidate(item['text'])
        if key not in seen:
            seen.add(key)
            out.append(item['text'])
    return out


def token_f1(a, b):
    a_toks, b_toks = word_tokenize(a.lower()), word_tokenize(b.lower())
    if not a_toks or not b_toks:
        return 0.0
    a_cnt, b_cnt = Counter(a_toks), Counter(b_toks)
    overlap = sum((a_cnt & b_cnt).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(a_toks)
    recall = overlap / len(b_toks)
    return 2 * precision * recall / (precision + recall)


def select_tokenf1_dedup(raw_items):
    texts = dedup_texts(raw_items)
    if len(texts) == 1:
        return texts[0]
    best_text, best_score = None, -1.0
    for i, ti in enumerate(texts):
        others = [tj for j, tj in enumerate(texts) if j != i]
        score = sum(token_f1(ti, tj) for tj in others) / len(others)
        if score > best_score:
            best_score, best_text = score, ti
    return best_text


def load_jsonl(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


pool = load_jsonl(CANDIDATE_POOL_PATH)
print(f'Loaded {len(pool)} rows from {CANDIDATE_POOL_PATH}')

rows = []
for r in pool:
    raw = r['raw_candidates'][:BUDGET]
    if r['predicted_type'] == 'passage' and len(raw) > 1:
        pred = select_tokenf1_dedup(raw)
    else:
        pred = raw[0]['text']
    if not pred or not pred.strip():
        pred = 'unknown'
    rows.append({'id': r['id'], 'spoiler': pred})

rows.sort(key=lambda r: r['id'])
with open(OUTPUT_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'spoiler'])
    for row in rows:
        writer.writerow([row['id'], row['spoiler']])

print(f'Wrote {len(rows)} predictions to {OUTPUT_PATH}')
print('This should be byte-identical to submission_task2_tokenf1_budget13.csv (the actual 0.45412 submission)')
