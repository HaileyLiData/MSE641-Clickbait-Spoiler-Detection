"""
================================================================================
Oracle Evidence Experiment
================================================================================

目的：验证evidence selection是否是瓶颈

方法：
1. 用gold answer找passage中最相似的句子（作弊，但用于分析）
2. 只把这些oracle句子给T5生成
3. 对比原始分数，看提升空间

结果解释：
- Oracle显著提升 → 瓶颈是evidence selection，值得做retriever
- Oracle也不提升 → 瓶颈在生成/数据/评测本身
- phrase提升小、passage提升大 → passage需要evidence selection

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
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
from google.colab import files
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm

# ================================================================================
# 配置
# ================================================================================
MODEL_PATH = '/content/drive/MyDrive/flan_t5_best_model'
SBERT_MODEL = 'all-MiniLM-L6-v2'
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128

print("=" * 70)
print("Oracle Evidence Experiment")
print("=" * 70)

# ================================================================================
# 上传数据
# ================================================================================
print("\n请上传 val.jsonl")
files.upload()

# ================================================================================
# 加载数据和模型
# ================================================================================
def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

val_raw = load_jsonl('/content/val.jsonl')
print(f"\n验证集样本数: {len(val_raw)}")

# 加载T5
print(f"\n加载T5: {MODEL_PATH}")
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
t5_model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t5_model.to(device)
t5_model.eval()
print(f"Device: {device}")

# 加载Sentence-BERT（用于找oracle evidence）
print(f"\n加载Sentence-BERT: {SBERT_MODEL}")
sbert = SentenceTransformer(SBERT_MODEL)
sbert.to(device)

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

def find_oracle_sentences(item, reference, top_k=2):
    """
    用reference找passage中最相似的句子（Oracle，作弊用于分析）
    """
    sentences = get_sentences(item)
    if not sentences:
        return ' '.join(item['targetParagraphs'])[:500]

    # 计算reference和每个句子的相似度
    ref_emb = sbert.encode(reference, convert_to_tensor=True)
    sent_embs = sbert.encode(sentences, convert_to_tensor=True)
    similarities = util.cos_sim(ref_emb, sent_embs)[0]

    # 取top-k
    k = min(top_k, len(sentences))
    top_indices = torch.argsort(similarities, descending=True)[:k]

    # 按原文顺序排列
    top_indices_sorted = sorted(top_indices.cpu().numpy())
    oracle_sentences = [sentences[i] for i in top_indices_sorted]

    return ' '.join(oracle_sentences)

def build_input_original(item):
    """原始输入格式"""
    spoiler_type = get_spoiler_type(item)
    question = ' '.join(item['postText'])
    context = item['targetTitle'] + ' ' + ' '.join(item['targetParagraphs'])
    return f"type: {spoiler_type} question: {question} context: {context}"

def build_input_oracle(item, oracle_evidence):
    """Oracle输入格式（用oracle evidence替换全文）"""
    spoiler_type = get_spoiler_type(item)
    question = ' '.join(item['postText'])
    context = item['targetTitle'] + ' ' + oracle_evidence
    return f"type: {spoiler_type} question: {question} context: {context}"

def predict(input_text):
    inputs = tokenizer(
        input_text,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        return_tensors='pt'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = t5_model.generate(
            **inputs,
            max_length=MAX_TARGET_LENGTH,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

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
# 实验1: 原始方法（baseline）
# ================================================================================
print("\n" + "=" * 70)
print("实验1: 原始方法 (Baseline)")
print("=" * 70)

preds_original = []
references = []
types = []

for item in tqdm(val_raw, desc="原始方法"):
    input_text = build_input_original(item)
    pred = predict(input_text)
    ref = get_spoiler_text(item)
    t = get_spoiler_type(item)

    preds_original.append(pred)
    references.append(ref)
    types.append(t)

meteor_original = compute_meteor(preds_original, references)
print(f"\n原始方法 METEOR: {meteor_original:.4f}")

# ================================================================================
# 实验2: Oracle Evidence (Top-1)
# ================================================================================
print("\n" + "=" * 70)
print("实验2: Oracle Evidence (Top-1句子)")
print("=" * 70)

preds_oracle_1 = []

for item, ref in tqdm(zip(val_raw, references), desc="Oracle Top-1", total=len(val_raw)):
    oracle_evidence = find_oracle_sentences(item, ref, top_k=1)
    input_text = build_input_oracle(item, oracle_evidence)
    pred = predict(input_text)
    preds_oracle_1.append(pred)

meteor_oracle_1 = compute_meteor(preds_oracle_1, references)
print(f"\nOracle Top-1 METEOR: {meteor_oracle_1:.4f} ({'+' if meteor_oracle_1 > meteor_original else ''}{meteor_oracle_1 - meteor_original:.4f})")

# ================================================================================
# 实验3: Oracle Evidence (Top-2)
# ================================================================================
print("\n" + "=" * 70)
print("实验3: Oracle Evidence (Top-2句子)")
print("=" * 70)

preds_oracle_2 = []

for item, ref in tqdm(zip(val_raw, references), desc="Oracle Top-2", total=len(val_raw)):
    oracle_evidence = find_oracle_sentences(item, ref, top_k=2)
    input_text = build_input_oracle(item, oracle_evidence)
    pred = predict(input_text)
    preds_oracle_2.append(pred)

meteor_oracle_2 = compute_meteor(preds_oracle_2, references)
print(f"\nOracle Top-2 METEOR: {meteor_oracle_2:.4f} ({'+' if meteor_oracle_2 > meteor_original else ''}{meteor_oracle_2 - meteor_original:.4f})")

# ================================================================================
# 实验4: Oracle Evidence (Top-3)
# ================================================================================
print("\n" + "=" * 70)
print("实验4: Oracle Evidence (Top-3句子)")
print("=" * 70)

preds_oracle_3 = []

for item, ref in tqdm(zip(val_raw, references), desc="Oracle Top-3", total=len(val_raw)):
    oracle_evidence = find_oracle_sentences(item, ref, top_k=3)
    input_text = build_input_oracle(item, oracle_evidence)
    pred = predict(input_text)
    preds_oracle_3.append(pred)

meteor_oracle_3 = compute_meteor(preds_oracle_3, references)
print(f"\nOracle Top-3 METEOR: {meteor_oracle_3:.4f} ({'+' if meteor_oracle_3 > meteor_original else ''}{meteor_oracle_3 - meteor_original:.4f})")

# ================================================================================
# 分类型分析
# ================================================================================
print("\n" + "=" * 70)
print("分类型分析")
print("=" * 70)

for t in ['phrase', 'passage', 'multi']:
    indices = [i for i, x in enumerate(types) if x == t]
    if not indices:
        continue

    t_original = [preds_original[i] for i in indices]
    t_oracle_1 = [preds_oracle_1[i] for i in indices]
    t_oracle_2 = [preds_oracle_2[i] for i in indices]
    t_oracle_3 = [preds_oracle_3[i] for i in indices]
    t_ref = [references[i] for i in indices]

    m_original = compute_meteor(t_original, t_ref)
    m_oracle_1 = compute_meteor(t_oracle_1, t_ref)
    m_oracle_2 = compute_meteor(t_oracle_2, t_ref)
    m_oracle_3 = compute_meteor(t_oracle_3, t_ref)

    print(f"\n{t} (n={len(indices)}):")
    print(f"  原始:       {m_original:.4f}")
    print(f"  Oracle-1:   {m_oracle_1:.4f} ({'+' if m_oracle_1 > m_original else ''}{m_oracle_1 - m_original:.4f})")
    print(f"  Oracle-2:   {m_oracle_2:.4f} ({'+' if m_oracle_2 > m_original else ''}{m_oracle_2 - m_original:.4f})")
    print(f"  Oracle-3:   {m_oracle_3:.4f} ({'+' if m_oracle_3 > m_original else ''}{m_oracle_3 - m_original:.4f})")

# ================================================================================
# 最终结果
# ================================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(f"\n{'方法':<15} {'METEOR':<10} {'变化':<10}")
print("-" * 40)
print(f"{'原始':<15} {meteor_original:.4f}")
print(f"{'Oracle-1':<15} {meteor_oracle_1:.4f}     {'+' if meteor_oracle_1 > meteor_original else ''}{meteor_oracle_1 - meteor_original:.4f}")
print(f"{'Oracle-2':<15} {meteor_oracle_2:.4f}     {'+' if meteor_oracle_2 > meteor_original else ''}{meteor_oracle_2 - meteor_original:.4f}")
print(f"{'Oracle-3':<15} {meteor_oracle_3:.4f}     {'+' if meteor_oracle_3 > meteor_original else ''}{meteor_oracle_3 - meteor_original:.4f}")

# ================================================================================
# 结论
# ================================================================================
print("\n" + "=" * 70)
print("结论")
print("=" * 70)

best_oracle = max(meteor_oracle_1, meteor_oracle_2, meteor_oracle_3)
improvement = best_oracle - meteor_original

if improvement > 0.05:
    print(f"\nOracle提升显著 (+{improvement:.4f})!")
    print("→ 瓶颈是evidence selection")
    print("→ 值得做sentence retriever")
elif improvement > 0.02:
    print(f"\nOracle有一定提升 (+{improvement:.4f})")
    print("→ evidence selection有帮助，但不是唯一瓶颈")
    print("→ 可以尝试retriever，但也要优化其他方面")
else:
    print(f"\nOracle提升有限 (+{improvement:.4f})")
    print("→ 瓶颈不在evidence selection")
    print("→ 问题在生成目标、数据噪声或评测本身")

print("=" * 70)
