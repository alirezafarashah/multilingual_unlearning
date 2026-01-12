# Multilingual Amnesia: On the Transferability of Unlearning in Multilingual LLMs

This is the **official repository** for the paper:

**Multilingual Amnesia: On the Transferability of Unlearning in Multilingual LLMs**  
Accepted at the **19th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2026)**

---

## 📄 Abstract

As multilingual large language models become more widely used, ensuring their safety and fairness across diverse linguistic contexts presents unique challenges. While existing research on machine unlearning has mainly focused on monolingual settings, typically English, multilingual environments introduce additional complexities due to cross-lingual knowledge transfer and biases embedded in both pretraining and fine-tuning data. In this work, we address the problem of multilingual unlearning using the Aya-Expanse 8B model under two settings: (1) data unlearning and (2) concept unlearning. We extend benchmarks for factual knowledge and stereotypes into ten languages through translation—English, French, Arabic, Japanese, Russian, Farsi, Korean, Hindi, Hebrew, and Indonesian—spanning five language families and varying resource levels. Our experiments show that unlearning in high-resource languages tends to be more stable, with asymmetric transfer observed between typologically related languages. Moreover, analysis of linguistic distances reveals that syntactic similarity is the most predictive factor of cross-lingual unlearning effects.


---

## 🧠 Background: TOFU Benchmark

This work builds upon **TOFU: Task of Fictitious Unlearning**, a benchmark designed to evaluate unlearning performance of large language models on realistic question–answering tasks. TOFU consists of QA pairs derived from synthetically generated autobiographies of fictitious authors, enabling controlled evaluation of forgetting and retention.

Original TOFU resources:
- Website: https://locuslab.github.io/tofu  
- Paper: http://arxiv.org/abs/2401.06121  
- GitHub: https://github.com/locuslab/tofu  

---

## 🌍 Multilingual Extension

We extend TOFU to **10 languages**:

- English, French, Arabic, Japanese, Russian  
- Farsi, Korean, Hindi, Hebrew, Indonesian  

📁 **All datasets for all languages are provided directly in the `dataset/` folder.**  


---

## ⚙️ Installation

```bash
conda create -n tofu python=3.10
conda activate tofu
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

---

## 📂 Repository Structure

```
.
├── config/
│   ├── finetune.yaml
│   ├── forget.yaml
│   ├── eval_everything.yaml
│   └── aggregate_eval_stat.yaml
├── dataset/
│   └── ...
├── finetune.py
├── forget.py
├── evaluate_util.py
├── aggregate_eval_stat.py
├── utils.py
└── seegull/
```

---

## 🚀 Workflow

### 1️⃣ Finetuning

First, fine-tune the model on the **full dataset**:

```bash
python finetune.py
```

You can modify training hyperparameters, model selection, and data paths in:
```
config/finetune.yaml
```

---

### 2️⃣ Unlearning 

To forget a specific dataset or language subset, run:

```bash
python forget.py
```

You can specify **which dataset or language to forget** by editing:
```
config/forget.yaml
```

---

### 3️⃣ Evaluation

To evaluate the unlearned models, run:

```bash
python evaluate_util.py
```

Evaluation settings, including **per-language evaluation**, can be configured in:
```
config/eval_everything.yaml
```

---

### 4️⃣ Aggregating Evaluation Results

To aggregate evaluation results and compute overall **model utility** and **forget quality**, run:

```bash
python aggregate_eval_stat.py
```

You can specify:
- the **retain model evaluation path**
- the **unlearned model evaluation path**

by editing:
```
config/aggregate_eval_stat.yaml
```

---



## 📝 Notes

- All experiments in the paper are conducted using **Aya-Expanse 8B**.
- To reproduce results related to **geo-cultural stereotype unlearning (SeeGULL)**, please refer to the README file in the `seegull/` directory.



---

## 📚 Citation

If you use this repository, please cite our paper:
```bibtex
@misc{farashah2026multilingualamnesiatransferabilityunlearning,
  title         = {Multilingual Amnesia: On the Transferability of Unlearning in Multilingual LLMs},
  author        = {Dehghanpour Farashah, Alireza and Khandelwal, Aditi and Fauchard, Marylou and Shi, Zhuan and Rostamzadeh, Negar and Farnadi, Golnoosh},
  year          = {2026},
  eprint        = {2601.05641},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2601.05641}
}
```

---

## 📬 Contact

For questions, issues, or collaborations, please open an issue or contact the authors.
