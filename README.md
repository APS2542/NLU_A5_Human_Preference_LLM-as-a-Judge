# A5: Optimization Human Preference & LLM-as-a-Judge

**Course:** AT82.05 Artificial Intelligence: Natural Language Understanding (NLU)

**Assignment:** A5 – Optimization Human Preference & LLM-as-a-Judge

**Student ID:** st126130

---

# Overview

This project explores **alignment and evaluation of Large Language Models (LLMs)** using **Direct Preference Optimization (DPO)** and **LLM-as-a-Judge evaluation**.

The goal is to fine-tune a pre-trained language model using **human preference data** and evaluate whether the fine-tuned model produces **more truthful and helpful responses**.

The evaluation compares the **baseline model** and the **DPO fine-tuned model** using a strong LLM judge on the AlpacaEval benchmark.

---

# Tasks

## Task 1 — Dataset Preparation

We use the dataset:

`tatsu-lab/alpaca_eval`

Training data comes from:

`jondurbin/truthy-dpo-v0.1`

Each training example contains:

* **prompt** – user instruction
* **chosen** – preferred truthful response
* **rejected** – incorrect or hallucinated response

The dataset is cleaned and filtered before training.

---

## Task 2 — DPO Training

The model is fine-tuned using **Direct Preference Optimization (DPO)**.

### Base Model

`Qwen/Qwen2.5-0.5B-Instruct`

### Training Method

* DPOTrainer (from TRL)
* LoRA for efficient fine-tuning
* Small subset of dataset for faster training

Training loss is recorded and visualized to monitor the optimization process.

---

## Task 3 — HuggingFace Model Upload

The trained model is uploaded to HuggingFace Hub.

Model Link:

https://huggingface.co/Aphisit-xt/qwen-dpo-model

This allows the fine-tuned model to be reused and evaluated easily.

---

## Task 4 — Evaluation with LLM-as-a-Judge

To evaluate model performance, we use the **AlpacaEval helpful_base benchmark**.

### Evaluation Process

1. Randomly sample **15 prompts** from AlpacaEval.
2. Generate responses from:

   * **Model A:** Base model
   * **Model B:** DPO fine-tuned model
3. Use **GPT-4o-mini as an automatic judge** to determine which response is better.

The judge outputs one of:

* Model A
* Model B
* Tie

---

# Results

| Metric           | Value   |
| ---------------- | ------- |
| Total Samples    | 15      |
| Model A Wins     | 2       |
| Model B Wins     | 5       |
| Ties             | 8       |
| **DPO Win Rate** | **60%** |

Win Rate Formula:

[
WinRate = \frac{ModelB + 0.5 \times Tie}{Total} \times 100
]

---

# Discussion

The evaluation results show that the **DPO-trained model wins more comparisons than the baseline model**.

However, a relatively large number of ties indicates that many responses are similar in quality. This is expected because:

* The training dataset is small
* The base model is already strong

Despite the limited training samples, the results suggest that **DPO improves response quality in several cases**.

---

# Technologies Used

* Python
* HuggingFace Transformers
* TRL (DPOTrainer)
* PEFT (LoRA)
* Datasets
* OpenAI API
* Google Colab

---

# References

* https://huggingface.co/docs/trl/main/dpo_trainer
* https://huggingface.co/datasets/jondurbin/truthy-dpo-v0.1
* https://huggingface.co/datasets/tatsu-lab/alpaca_eval

---
