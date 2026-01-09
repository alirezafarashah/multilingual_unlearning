

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Workflow

### 1️⃣ Unlearn Biases
Train a model to unlearn geo-cultural stereotypes:
```bash
python main.py \
  --model_name unsloth/Meta-Llama-3.1-8B-Instruct \
  --learning_rate 1e-6 \
  --weights 1.0 0.25 0.5 \
  --dataset ./mcq_stereotype_dataset.csv \
  --language en
```

Replace `en` with any of the supported languages. Use corresponding dataset as well.

---

### 2️⃣ Evaluate (MCQ probabilities)
Compute per-option (and per-letter) probabilities on the MCQ dataset:
```bash
python eval_prob.py \
  --model_dir ./saved_models/ \
  --dataset ./mcq_stereotype_dataset.csv \
  --output_dir ./outputs/ \
  --filename unlearned_models_mcq_probs \
  --language en
```
---

