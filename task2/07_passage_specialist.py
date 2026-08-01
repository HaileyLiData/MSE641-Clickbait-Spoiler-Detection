"""
================================================================================
Step2: Passage Specialist 训练 (修复版)
================================================================================
修复: fp16 → bf16，避免数值溢出导致 optimizer step 被跳过

问题诊断:
- 原版 fp16=True 导致 NaN/Inf
- GradScaler 跳过 optimizer.step()
- 权重完全不更新
- Trainer 日志显示 0.000000

修复方案:
- bf16=True (A100 更适合)
- logging_nan_inf_filter=False (暴露问题)
- 训练前后权重检查

先用 max_steps=20 诊断，确认正常后再完整训练
================================================================================
"""

import json
import torch
import random
import numpy as np
import math
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from datasets import Dataset
from google.colab import drive, files

drive.mount('/content/drive')

# ============== 配置 ==============
SEED = 123
BASE_MODEL_PATH = '/content/drive/MyDrive/flan_t5_large_seed123/model'
OUTPUT_DIR = '/content/drive/MyDrive/flan_t5_large_seed123_passage_specialist/model'
DATA_DIR = '/content/drive/MyDrive/Clickbait data'
TRAIN_FILE = f'{DATA_DIR}/train.jsonl'

MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 128

# 训练参数 (与原始Seed123一致)
NUM_EPOCHS = 5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16
LEARNING_RATE = 5e-5
WARMUP_STEPS = 20  # 约5%
WEIGHT_DECAY = 0.01

# 诊断模式: 已确认正常，完整训练
DIAGNOSTIC_MODE = False
MAX_STEPS_DIAGNOSTIC = 20


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_spoiler_type(row):
    tags = row.get('tags', ['phrase'])
    return tags[0] if isinstance(tags, list) and tags else 'phrase'


def build_input(row):
    """构建输入 - passage specialist 输入固定为 type: passage"""
    post = ' '.join(row['postText'])
    title = row['targetTitle']
    paragraphs = ' '.join(row['targetParagraphs'])
    return f'type: passage question: {post} context: {title} {paragraphs}'


def load_passage_data(path):
    """只加载 passage 类型的训练数据"""
    inputs, targets = [], []
    total, passage_count = 0, 0

    with open(path) as f:
        for line in f:
            d = json.loads(line)
            total += 1
            spoiler_type = get_spoiler_type(d)

            if spoiler_type == 'passage':
                passage_count += 1
                inputs.append(build_input(d))
                targets.append(' '.join(d['spoiler']))

    print(f"总样本数: {total}")
    print(f"Passage样本数: {passage_count} ({passage_count/total*100:.1f}%)")

    return Dataset.from_dict({'input': inputs, 'target': targets})


def preprocess(examples, tokenizer):
    model_inputs = tokenizer(
        examples['input'],
        max_length=MAX_INPUT_LEN,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        examples['target'],
        max_length=MAX_TARGET_LEN,
        truncation=True,
        padding=False,
    )
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


# 诊断回调
class LossCheckCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return

        loss = logs["loss"]
        print(f"[诊断] step={state.global_step}, loss={loss}")

        if not math.isfinite(loss):
            raise RuntimeError(f"检测到非有限 loss: {loss}，停止训练。")


def main():
    print("=" * 70)
    print("Seed123 Passage Specialist 训练 (修复版)")
    print("=" * 70)

    if DIAGNOSTIC_MODE:
        print("\n⚠️ 诊断模式: 只跑 20 步，确认训练正常")

    set_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 加载数据
    print(f"\n加载训练数据: {TRAIN_FILE}")
    train_dataset = load_passage_data(TRAIN_FILE)
    print(f"训练样本数: {len(train_dataset)}")

    # 加载模型
    print(f"\n加载基础模型: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_PATH)

    # 修复 tied weights warning
    model.config.tie_word_embeddings = False

    # 预处理
    print("\n预处理数据...")
    train_dataset = train_dataset.map(
        lambda x: preprocess(x, tokenizer),
        batched=True,
        remove_columns=['input', 'target'],
    )

    # 训练参数 (修复版)
    training_args = Seq2SeqTrainingArguments(
        output_dir='/content/passage_specialist_checkpoints',
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        optim='adafactor',

        # 日志设置
        logging_strategy='steps',
        logging_steps=10,
        logging_first_step=True,
        logging_nan_inf_filter=False,  # 关键：暴露 NaN/Inf

        save_strategy='epoch',
        save_total_limit=2,

        # 修复：用 bf16 替代 fp16
        fp16=False,
        bf16=True,  # A100 更适合
        tf32=True,

        report_to='none',
        seed=SEED,

        # 诊断模式
        max_steps=MAX_STEPS_DIAGNOSTIC if DIAGNOSTIC_MODE else -1,
    )

    # 训练
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[LossCheckCallback()],
    )

    # 训练前权重快照
    probe_name = "encoder.block.0.layer.0.SelfAttention.q.weight"
    before = dict(model.named_parameters())[probe_name].detach().float().cpu().clone()

    print("\n开始训练...")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION}")
    print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  BF16: True (修复)")
    print(f"  诊断模式: {DIAGNOSTIC_MODE}")

    train_result = trainer.train()

    # 训练后权重检查
    after = dict(trainer.model.named_parameters())[probe_name].detach().float().cpu()
    max_diff = (after - before).abs().max().item()

    print("\n" + "=" * 50)
    print("权重更新检查")
    print("=" * 50)
    print(f"训练后内存权重最大差异: {max_diff}")
    print(f"训练指标: {train_result.metrics}")

    if max_diff > 0:
        print("✅ 权重已更新，训练正常!")
    else:
        print("❌ 权重未更新，仍有问题!")
        return

    # 诊断模式不保存
    if DIAGNOSTIC_MODE:
        print("\n⚠️ 诊断模式完成，权重已更新")
        print("请将 DIAGNOSTIC_MODE 改为 False，重新完整训练")
        return

    # 保存模型
    print(f"\n保存模型到: {OUTPUT_DIR}")
    trainer.model.config.tie_word_embeddings = False
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("训练完成!")
    print(f"模型已保存: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
