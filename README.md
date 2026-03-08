# A5: Optimization Human Preference & LLM-as-a-Judge

**Course:** AT82.05 Artificial Intelligence: Natural Language Understanding (NLU)

**Assignment:** A5 – Optimization Human Preference & LLM-as-a-Judge

**Student ID:** st126130

---
## Files Structure

This repository contains the notebook, exported report, and output files generated during the assignment.

```
.
├── A5_notebook_st126130.ipynb      # Main notebook
├── A5_notebook_st126130.pdf        # Notebook export        
├── README.md                       # Project description
├── judge_results.csv               # Judging results
├── model_outputs.csv               # Model outputs
└── task4_summary.csv               # Task 4 summary  
```
---

## Project Overview

This project explores **alignment and evaluation of Large Language Models (LLMs)** using:

* **Direct Preference Optimization (DPO)** for model alignment
* **LLM-as-a-Judge** for automatic evaluation

The objective is to fine-tune a pre-trained language model to produce **more truthful and helpful responses**, and then evaluate whether the fine-tuned model improves over the original base model.

---

## Project Pipeline

```
Human Preference Dataset
        │
        │
        ▼
Dataset Cleaning & Preparation
(prompt, chosen, rejected)
        │
        ▼
DPO Training
(Base Model → Fine-tuned Model)
        │
        ▼
Model Upload to HuggingFace
        │
        ▼
Evaluation with AlpacaEval
        │
        ▼
LLM Judge (GPT-4o-mini)
        │
        ▼
Win Rate Calculation
```

---

## Task 1 — Dataset Preparation

We use a preference dataset designed to improve model truthfulness.

Dataset:

```
jondurbin/truthy-dpo-v0.1
```

Each data sample contains:

| Field    | Description                        |
| -------- | ---------------------------------- |
| prompt   | user instruction                   |
| chosen   | preferred truthful response        |
| rejected | incorrect or hallucinated response |

The dataset is:

* cleaned
* filtered for valid rows
* formatted for DPO training

---

## Task 2 — Model Training with DPO

We fine-tune a pre-trained language model using **Direct Preference Optimization (DPO)**.

### Base Model

```
Qwen/Qwen2.5-0.5B-Instruct
```

### Training Setup

| Parameter          | Value      |
| ------------------ | ---------- |
| Training Method    | DPOTrainer |
| Fine-tuning Method | LoRA       |
| Training Samples   | 10         |
| Epochs             | 1          |
| Learning Rate      | 5e-6       |

LoRA is used to reduce memory usage and make training feasible in Google Colab.

### Training Flow

```
Prompt
  │
  ├── Chosen Response (preferred)
  │
  └── Rejected Response (incorrect)
          │
          ▼
DPO Optimization
          │
          ▼
Model learns to prefer chosen responses
```

### Training Loss

Training loss is logged during optimization and visualized to monitor the training process.

---

## Task 3 — HuggingFace Model Upload

After training, the fine-tuned model is uploaded to HuggingFace Hub.

Model Link:

```
https://huggingface.co/Aphisit-xt/qwen-dpo-model
```

This allows the trained model to be reused and evaluated easily.

---

## Task 4 — Evaluation with LLM-as-a-Judge

To evaluate model performance, we use the **AlpacaEval benchmark**.

Dataset:

```
tatsu-lab/alpaca_eval
```

Subset used:

```
helpful_base
```

### Evaluation Process

1. Randomly sample **15 prompts**
2. Generate responses using:

   * **Model A:** Base model
   * **Model B:** DPO fine-tuned model
3. Send both responses to an **LLM judge**
4. The judge selects the better answer

Possible outputs:

```
Model A
Model B
Tie
```

---

## Evaluation Pipeline

```
Prompt
   │
   ├── Base Model (Model A)
   │        │
   │        └── Response A
   │
   └── DPO Model (Model B)
            │
            └── Response B
                    │
                    ▼
             LLM Judge
           (GPT-4o-mini)
                    │
                    ▼
                Verdict
```

---

## Results

| Metric           | Value   |
| ---------------- | ------- |
| Total Samples    | 15      |
| Model A Wins     | 2       |
| Model B Wins     | 5       |
| Ties             | 8       |
| **DPO Win Rate** | **60%** |

### Win Rate Formula

```
Win Rate =
(Model B Wins + 0.5 × Ties) / Total Evaluations × 100
```

---

## Discussion

The results show that the **DPO-trained model wins more comparisons than the baseline model**.

However, many responses result in ties, which suggests that:

* The base model is already strong
* The training dataset is relatively small
* Limited training samples reduce the magnitude of improvement

Despite the small training size, the results suggest that **DPO can improve response quality in some cases**.

---

## Technologies Used

| Tool         | Purpose                   |
| ------------ | ------------------------- |
| Python       | main programming language |
| Transformers | loading and running LLMs  |
| TRL          | DPO training              |
| PEFT (LoRA)  | efficient fine-tuning     |
| Datasets     | dataset loading           |
| OpenAI API   | LLM judge                 |
| Google Colab | training environment      |

---

## References

DPO Trainer
https://huggingface.co/docs/trl/main/dpo_trainer

Truthful DPO Dataset
https://huggingface.co/datasets/jondurbin/truthy-dpo-v0.1

AlpacaEval Benchmark
https://huggingface.co/datasets/tatsu-lab/alpaca_eval

---

