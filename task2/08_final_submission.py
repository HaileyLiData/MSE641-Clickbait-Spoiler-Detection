"""
================================================================================
Step 2: 从候选池生成提交文件
================================================================================
方法: All-F1 (Token-F1 consensus)

输入:
- test_seed123_passage_specialist_candidate_pool.jsonl (候选池)

输出:
- task2_submission.csv (提交文件)
================================================================================
"""

import json
import csv
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from google.colab import drive, files

nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

drive.mount('/content/drive')

# ============== 配置 ==============
DATA_DIR = '/content/drive/MyDrive/Clickbait data'
POOL_FILE = f'{DATA_DIR}/test_seed123_passage_specialist_candidate_pool.jsonl'
OUTPUT_FILE = f'{DATA_DIR}/task2_submission.csv'


# ============== 筛选函数 (All-F1) ==============
def tok(s):
    return word_tokenize(s.lower())


def normalize(t):
    return ' '.join(t.lower().split())


def dedup_texts(raw_items):
    seen, out = set(), []
    for item in raw_items:
        text = item['text'] if isinstance(item, dict) else item
        k = normalize(text)
        if k not in seen:
            seen.add(k)
            out.append(text)
    return out


def token_f1(a, b):
    a_t, b_t = tok(a), tok(b)
    if not a_t or not b_t:
        return 0.0
    a_c, b_c = Counter(a_t), Counter(b_t)
    ov = sum((a_c & b_c).values())
    if ov == 0:
        return 0.0
    p, r = ov / len(a_t), ov / len(b_t)
    return 2 * p * r / (p + r)


def consensus_select(texts, sim_fn):
    if len(texts) == 1:
        return texts[0]
    best_t, best_s = None, -1.0
    for i, ti in enumerate(texts):
        others = [tj for j, tj in enumerate(texts) if j != i]
        s = sum(sim_fn(ti, tj) for tj in others) / len(others)
        if s > best_s:
            best_s, best_t = s, ti
    return best_t


def select_all_f1(candidates):
    """All-F1方法: 用Token-F1 consensus筛选"""
    if len(candidates) > 1:
        unique = dedup_texts(candidates)
        if len(unique) == 1:
            return unique[0]
        return consensus_select(unique, token_f1)
    else:
        text = candidates[0]['text'] if isinstance(candidates[0], dict) else candidates[0]
        return text


def main():
    print("=" * 60)
    print("Step 2: 从候选池生成提交文件")
    print("=" * 60)
    print(f"方法: All-F1 (Token-F1 consensus)")

    # 加载候选池
    print(f"\n加载候选池: {POOL_FILE}")
    pool = []
    with open(POOL_FILE) as f:
        for line in f:
            pool.append(json.loads(line))
    print(f"样本数: {len(pool)}")

    # 筛选
    print(f"\n开始筛选...")
    results = []
    for i, record in enumerate(pool):
        candidates = record['raw_candidates']
        spoiler = select_all_f1(candidates)
        results.append((i, spoiler))

    # 保存提交文件 (CSV格式: id,spoiler)
    with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'spoiler'])
        for idx, spoiler in results:
            writer.writerow([idx, spoiler])

    print(f"\n提交文件已保存: {OUTPUT_FILE}")
    print(f"共 {len(results)} 条预测")

    # 下载提交文件
    files.download(OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("完成！可以提交到 CodaLab")
    print("=" * 60)


if __name__ == '__main__':
    main()
