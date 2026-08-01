# MSE641 Clickbait Spoiler Detection

This repository contains the code for our MSE641 Text Analytics course project at the University of Waterloo (Spring 2026).

## Project Overview

We participated in the **Webis Clickbait Spoiler Detection** shared task with two subtasks:
- **Task 1**: Spoiler Type Classification (phrase / passage / multi)
- **Task 2**: Spoiler Generation

## Results

| Task | Method | Kaggle Score |
|------|--------|--------------|
| Task 1 | DeBERTa-v1-base + FGM + Class Weights | **0.77304** |
| Task 2 | Flan-T5-large + Multi-candidate + Token-F1 | **0.47303** |

## Repository Structure

```
.
├── README.md
├── task1/                              # Classification Task
│   ├── 01_baseline_deberta_v1.py       # Baseline (0.765)
│   ├── 02_model_scale_v3_large.py      # Model scale comparison
│   └── 03_final_fgm_classweight.py     # Final model (0.773)
│
└── task2/                              # Generation Task
    ├── 01_teacher_baseline_qa.py       # Teacher baseline (0.3577)
    ├── 02_generator_exploration/       # Model exploration
    │   ├── t5_base.py
    │   ├── bart.py
    │   └── flant5_large.py             # Best generator (0.4124)
    ├── 03_type_conditioned.py          # Type-conditioned prompts
    ├── 04_oracle_evidence_test.py      # Oracle experiments (key finding)
    ├── 05_sentence_retriever.py        # Learned retrieval (negative result)
    ├── 06_multi_candidate_mbr.py       # Multi-candidate selection
    ├── 07_passage_specialist.py        # Passage specialist training
    └── 08_final_submission.py          # Final pipeline (0.473)
```

## Key Findings

### Task 1: Classification
- **Smaller models can outperform larger ones**: DeBERTa-v1-base (86M) achieved better results than DeBERTa-v3-large (304M)
- **FGM adversarial training** + **class weighting** + **cosine scheduler** improved robustness

### Task 2: Generation
- **Oracle evidence experiments** revealed that evidence selection is the bottleneck (+0.27 METEOR with oracle evidence for passage type)
- **Multi-candidate generation** with **Token-F1 consensus selection** outperformed single-candidate approaches
- **Passage specialist** (continuation fine-tuning) further improved performance

## Requirements

```
torch>=1.10
transformers>=4.20
datasets
scikit-learn
nltk
```

## Usage

Most scripts are designed for Google Colab with GPU. Key hyperparameters:
- Task 1: `max_length=256`, `batch_size=16`, `epochs=5`
- Task 2: `max_length=512`, `batch_size=8`, `epochs=3`

## Team

- Hailey - Task 1 fine-tuning, Task 2 oracle experiments & final system
- Rocha - Task 1 baseline, Task 2 generator exploration & MBR selection

## License

For academic use only. University of Waterloo, MSE641 Spring 2026.
