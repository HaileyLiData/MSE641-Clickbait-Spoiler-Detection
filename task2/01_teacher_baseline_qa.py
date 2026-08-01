#!/usr/bin/env python3
"""
================================================================================
MSE 641 - Task 2 Transformer Baseline (DeBERTa-large QA Model)
================================================================================
任务说明 / Task Description:
  - 这是一个抽取式问答任务 (Extractive Question Answering)
  - 把clickbait标题作为"问题"
  - 把文章内容作为"上下文"
  - 模型从上下文中抽取答案（即spoiler）

模型信息 / Model Info:
  - 来源: webis/clickbait-spoiling-with-question-answering
  - 分支: debertalarge-all-cbs20-both-checkpoint-1200
  - 类型: DeBERTa-large for Question Answering

输出 / Output:
  - submission_task2_transformer.csv (id, spoiler)
================================================================================
"""

import os
import json
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# ============================================================================
# 路径设置 / Path Configuration
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FILE = os.path.join(SCRIPT_DIR, "test.jsonl")
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")  # 本地模型目录
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "submission_task2_transformer.csv")

# 模型信息（从Hugging Face下载）/ Model info (download from HuggingFace)
MODEL_NAME = "webis/clickbait-spoiling-with-question-answering"
MODEL_REVISION = "debertalarge-all-cbs20-both-checkpoint-1200"

print("=" * 60)
print("MSE 641 - Task 2 Transformer Baseline")
print("=" * 60)

# ============================================================================
# 检查GPU / Check GPU
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# 加载数据 / Load Data
# ============================================================================
print("\nLoading test data...")

if not os.path.exists(TEST_FILE):
    print(f"ERROR: Cannot find {TEST_FILE}")
    exit(1)

test_data = []
with open(TEST_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        test_data.append(json.loads(line))

print(f"  {len(test_data)} samples loaded")

# ============================================================================
# 数据预处理 / Data Preprocessing
# ============================================================================
# 转换为QA格式:
# - question: clickbait标题 (postText)
# - context: 文章标题 + 文章内容 (targetTitle + targetParagraphs)
# Convert to QA format:
# - question: clickbait title (postText)
# - context: article title + content (targetTitle + targetParagraphs)

qa_data = []
for item in test_data:
    # 获取ID (兼容'id'和'uuid')
    sample_id = item.get('id', item.get('uuid'))

    # 问题 = clickbait标题
    question = ' '.join(item['postText'])

    # 上下文 = 文章标题 + 文章段落
    context = item['targetTitle'] + ' - ' + ' '.join(item['targetParagraphs'])

    qa_data.append({
        'id': sample_id,
        'question': question,
        'context': context
    })

print(f"  Data preprocessed into QA format")

# ============================================================================
# 加载/下载模型 / Load/Download Model
# ============================================================================
print(f"\nLoading QA model...")

# 检查本地模型是否存在 / Check if local model exists
if os.path.exists(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, 'config.json')):
    print(f"  Loading from local: {MODEL_DIR}")
    model_path = MODEL_DIR
else:
    print(f"  Downloading from HuggingFace: {MODEL_NAME}")
    print(f"  Revision: {MODEL_REVISION}")
    print("  This may take a few minutes for first run...")

    # 下载模型到本地 / Download model to local
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(
        MODEL_NAME,
        revision=MODEL_REVISION,
        local_dir=MODEL_DIR
    )
    print(f"  Model downloaded to: {model_path}")

# 加载tokenizer和模型 / Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)
model.to(DEVICE)
model.eval()
print("  Model loaded successfully!")

# ============================================================================
# 运行预测 / Run Predictions
# ============================================================================
print(f"\nRunning predictions on {len(qa_data)} samples...")

# QA预测函数 / QA prediction function
def predict_answer(question, context, max_length=512):
    """
    从上下文中抽取答案
    Extract answer from context using QA model

    参数 / Parameters:
        question: 问题（clickbait标题）
        context: 上下文（文章内容）
        max_length: 最大序列长度

    返回 / Returns:
        answer: 抽取的答案（spoiler）
    """
    # 编码输入 / Encode inputs
    inputs = tokenizer(
        question,
        context,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # 模型预测 / Model prediction
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取答案的开始和结束位置 / Get start and end positions
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # 找到最可能的答案位置 / Find most likely answer position
    start_idx = torch.argmax(start_logits, dim=1).item()
    end_idx = torch.argmax(end_logits, dim=1).item()

    # 确保end >= start / Ensure end >= start
    if end_idx < start_idx:
        end_idx = start_idx

    # 解码答案 / Decode answer
    input_ids = inputs['input_ids'][0]
    answer_tokens = input_ids[start_idx:end_idx + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    # 清理答案 / Clean answer
    answer = answer.strip()
    if not answer:
        # 如果答案为空，使用文章标题作为后备 / Fallback to title if empty
        answer = context.split(' - ')[0] if ' - ' in context else context[:100]

    return answer

# 批量预测 / Batch predictions
results = []
for item in tqdm(qa_data, desc="Predicting"):
    answer = predict_answer(item['question'], item['context'])
    results.append({
        'id': item['id'],
        'spoiler': answer
    })

print("  Predictions completed!")

# ============================================================================
# 保存结果 / Save Results
# ============================================================================
print(f"\nSaving results...")

# 保存为CSV (Kaggle格式) / Save as CSV (Kaggle format)
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"  CSV saved: {OUTPUT_CSV}")

# 显示示例 / Show examples
print("\nSample predictions:")
print("-" * 60)
for i in range(min(3, len(results))):
    print(f"ID: {results[i]['id']}")
    print(f"Spoiler: {results[i]['spoiler'][:100]}...")
    print("-" * 60)

print("\n" + "=" * 60)
print("Done! Submit to Kaggle:")
print(f"  {OUTPUT_CSV}")
print("=" * 60)
