import argparse
import os
import csv
import json
import random
from typing import List, Dict, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


LANG_SPEC = {
    "en": {"Q": "Question: ", "A": "Answer: ", "Q_END": " \n",
           "UNK": ["Cannot be determined", "Unknown", "Not enough information", "Unclear"]},
    "fr": {"Q": "Question: ", "A": "Réponse: ", "Q_END": " \n",
           "UNK": ["Impossible à déterminer", "Inconnu", "Pas assez d'informations", "Peu clair"]},
    "ar": {"Q": "السؤال: ", "A": "الإجابة: ", "Q_END": " \n",
           "UNK": ["لا يمكن تحديده", "غير معروف", "لا توجد معلومات كافية", "غير واضح"]},
    "fa": {"Q": "سوال: ", "A": "پاسخ: ", "Q_END": " \n",
           "UNK": ["قابل تعیین نیست", "نامشخص", "اطلاعات کافی وجود ندارد", "مبهم"]},
    "hi": {"Q": "प्रश्न: ", "A": "उत्तर: ", "Q_END": " \n",
           "UNK": ["निर्धारित नहीं किया जा सकता", "अज्ञात", "पर्याप्त जानकारी नहीं है", "अस्पष्ट"]},
    "iw": {"Q": "שאלה: ", "A": "תשובה: ", "Q_END": " \n",
           "UNK": ["לא ניתן לקבוע", "לא ידוע", "אין מספיק מידע", "לא ברור"]},
    "id": {"Q": "Pertanyaan: ", "A": "Jawaban: ", "Q_END": " \n",
           "UNK": ["Tidak dapat ditentukan", "Tidak diketahui", "Informasi tidak cukup", "Tidak jelas"]},
    "ja": {"Q": "質問: ", "A": "回答: ", "Q_END": " \n",
           "UNK": ["判断不能", "不明", "情報が不十分", "不明確"]},
    "ko": {"Q": "질문: ", "A": "답변: ", "Q_END": " \n",
           "UNK": ["판단할 수 없음", "알 수 없음", "정보가 충분하지 않음", "불분명"]},
    "ru": {"Q": "Вопрос: ", "A": "Ответ: ", "Q_END": " \n",
           "UNK": ["Невозможно определить", "Неизвестно", "Недостаточно информации", "Неясно"]},
}
DEFAULT_LANG = "en"

LETTERS = ["A", "B", "C", "D", "E"]

# Localized prefix for "correct answer is X" in each language (X in A..E)


def _lang(key: str, language: str) -> str:
    spec = LANG_SPEC.get(language, LANG_SPEC[DEFAULT_LANG])
    return spec[key]


def _unknown(language: str) -> str:
    # Choose one canonical Unknown text (index 2 by default)
    return LANG_SPEC.get(language, LANG_SPEC[DEFAULT_LANG])["UNK"][1]


def _softmax_from_logprobs(logps: List[float]) -> List[float]:
    a = np.array(logps, dtype=np.float64)
    a -= a.max()
    e = np.exp(a)
    p = e / e.sum()
    return p.tolist()


def load_model(save_directory: str):
    model = AutoModelForCausalLM.from_pretrained(save_directory, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(save_directory)
    print("Model and tokenizer loaded successfully!")
    return model, tokenizer


def save_csv_file(results: Dict[str, List], output_dir='./outputs/', filename='unlearned_models_mcq_probs'):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename + '.csv')
    with open(path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(results.keys())
        writer.writerows(zip(*results.values()))
    print(f"CSV saved to {path}")


def save_json_file(results: Dict[str, List], output_dir='./outputs/', filename='unlearned_models_mcq_probs'):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename + '.json')
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"JSON saved to {path}")


def _seq_logprob(model, tokenizer, context_text: str, option_text: str, device: str) -> float:
    """
    Sum of token log-probs for the full option continuation given the context:
    log P(option_tokens | context_text)
    """
    # Leading space makes option a natural continuation for most tokenizers
    cont = option_text if option_text.startswith(" ") else " " + option_text

    with torch.no_grad():
        ctx_ids = tokenizer(context_text, return_tensors="pt", add_special_tokens=False).to(device)
        full_ids = tokenizer(context_text + cont, return_tensors="pt", add_special_tokens=False).to(device)

        outputs = model(**full_ids)
        logits = outputs.logits[0]                     # [seq_len, vocab]
        log_probs = torch.log_softmax(logits, dim=-1)  # [seq_len, vocab]
        full = full_ids.input_ids[0]                   # [seq_len]

        start = ctx_ids.input_ids[0].size(0)
        end = full.size(0)

        total = 0.0
        for t in range(start, end):
            prev = t - 1
            tok_id = full[t].item()
            total += log_probs[prev, tok_id].item()
        return float(total)


def _format_options_block(options: List[str]) -> str:
    """Create a 5-line block:
    A) opt1
    B) opt2
    C) opt3
    D) opt4
    E) opt5
    """
    lines = [f"{lab}) {opt}" for lab, opt in zip(LETTERS, options)]
    return "\n".join(lines) + "\n"


def _shuffle_and_map(
    opts: List[str], unknown_text: str
) -> Tuple[List[Tuple[str, str]], Dict[int, str], Dict[str, int]]:
    """
    Build a list of (orig_label, text) for the 5 options, shuffle them,
    and return:
      - shuffled: List[(label, text)] in displayed order A..E
      - pos_to_label: {pos -> label} where label in {"Option_1".., "Unknown"}
      - label_to_pos: inverse map
    """
    items = [
        ("Option_1", opts[0]),
        ("Option_2", opts[1]),
        ("Option_3", opts[2]),
        ("Option_4", opts[3]),
        ("Unknown", unknown_text),
    ]
    shuffled = items[:]
    random.shuffle(shuffled)
    pos_to_label = {i: shuffled[i][0] for i in range(5)}
    label_to_pos = {label: pos for pos, label in pos_to_label.items()}
    return shuffled, pos_to_label, label_to_pos



# --- NEW/CHANGED IMPORTS (unchanged otherwise) ---
# (no new imports needed)

# --- CHANGED: evaluate_model signature and internals ---
def evaluate_model(model, tokenizer, dataset, output_dir='./outputs/', filename='unlearned_models_mcq_probs',
                   language='en', device="cuda:0", seed: int = 0,
                   unknown_text: str = None, unknown_id: int = None):
    random.seed(seed)
    model.to(device).eval()

    # Results schema
    results = {
        "language": [],
        "model_dir": [],
        "mcq_question": [],
        # Track which Unknown variant we used
        "unknown_id": [],
        "unknown_text": [],

        # Original (unshuffled) options and the (single) Unknown text used
        "option_1": [], "option_2": [], "option_3": [], "option_4": [], "unknown_option": [],
        "correct_option": [],

        # What the model SAW (shuffled) under A..E
        "presented_A": [], "presented_B": [], "presented_C": [], "presented_D": [], "presented_E": [],
        "label_A": [], "label_B": [], "label_C": [], "label_D": [], "label_E": [],
        "correct_letter": [], "unknown_letter": [],

        # raw log-probs by ORIGINAL labels (remapped from letters)
        "logp_option_1": [], "logp_option_2": [], "logp_option_3": [], "logp_option_4": [], "logp_unknown": [],
        # normalized probs by ORIGINAL labels (5-way softmax over option texts)
        "p_option_1": [], "p_option_2": [], "p_option_3": [], "p_option_4": [], "p_unknown": [],

        # Per-letter probabilities
        "p_choice_A": [], "p_choice_B": [], "p_choice_C": [], "p_choice_D": [], "p_choice_E": [],

        # Phrase-based probabilities
        "p_phrase_A": [], "p_phrase_B": [], "p_phrase_C": [], "p_phrase_D": [], "p_phrase_E": [],

        # Predictions
        "predicted_label": [], "predicted_text": [], "predicted_letter": [], "is_correct": [],
        "predicted_letter_phrase": [],
    }

    if language not in LANG_SPEC:
        raise ValueError(f"Unsupported language '{language}'. Supported: {list(LANG_SPEC.keys())}")

    Q, A, Q_END = _lang("Q", language), _lang("A", language), _lang("Q_END", language)
    unk = unknown_text if unknown_text is not None else _unknown(language)

    for sample in tqdm(dataset, desc="Evaluating"):
        q_text = sample["MCQ_Question"]
        opts = [sample["Option_1"], sample["Option_2"], sample["Option_3"], sample["Option_4"]]
        correct_label = sample["Correct_Option"]  # "Option_1".."Option_4"

        # Shuffle & map
        shuffled, pos_to_label, label_to_pos = _shuffle_and_map(opts, unk)
        displayed_texts = [text for (_label, text) in shuffled]  # A..E texts

        # Build context
        options_block = _format_options_block(displayed_texts)
        context = f"{Q}{q_text}{Q_END}{options_block}{A}"

        # --- Text-choice scoring ---
        logps_by_letter = []
        for text in displayed_texts:
            lp = _seq_logprob(model, tokenizer, context, text, device)
            logps_by_letter.append(lp)
        probs_by_letter = _softmax_from_logprobs(logps_by_letter)

        # Prediction based on text-choice
        pred_pos = int(np.argmax(probs_by_letter))
        pred_letter = LETTERS[pred_pos]
        pred_label = pos_to_label[pred_pos]
        pred_text = displayed_texts[pred_pos]
        is_correct = (pred_label == correct_label)

        # Remap indices
        idx_o1 = label_to_pos["Option_1"]
        idx_o2 = label_to_pos["Option_2"]
        idx_o3 = label_to_pos["Option_3"]
        idx_o4 = label_to_pos["Option_4"]
        idx_u  = label_to_pos["Unknown"]

        # --- Phrase-based scoring over letters A..E ---
        phrase_logps = []
        for L in LETTERS:
            phrase = f" {L}"  # (kept your current simplified phrase continuation)
            lp = _seq_logprob(model, tokenizer, context, phrase, device)
            phrase_logps.append(lp)
        phrase_probs = _softmax_from_logprobs(phrase_logps)
        pred_pos_phrase = int(np.argmax(phrase_probs))
        pred_letter_phrase = LETTERS[pred_pos_phrase]

        # Correct/unknown letters after shuffle
        correct_pos = label_to_pos[correct_label]
        correct_letter = LETTERS[correct_pos]
        unknown_pos = idx_u
        unknown_letter = LETTERS[unknown_pos]

        # Save row
        results["language"].append(language)
        results["model_dir"].append(getattr(model, "name_or_path", ""))
        results["mcq_question"].append(q_text)
        results["unknown_id"].append(unknown_id if unknown_id is not None else -1)
        results["unknown_text"].append(unk)

        # Original (unshuffled)
        results["option_1"].append(opts[0]); results["option_2"].append(opts[1])
        results["option_3"].append(opts[2]); results["option_4"].append(opts[3])
        results["unknown_option"].append(unk)
        results["correct_option"].append(correct_label)

        # Presented A..E
        results["presented_A"].append(displayed_texts[0]); results["presented_B"].append(displayed_texts[1])
        results["presented_C"].append(displayed_texts[2]); results["presented_D"].append(displayed_texts[3])
        results["presented_E"].append(displayed_texts[4])
        results["label_A"].append(pos_to_label[0]); results["label_B"].append(pos_to_label[1])
        results["label_C"].append(pos_to_label[2]); results["label_D"].append(pos_to_label[3])
        results["label_E"].append(pos_to_label[4])
        results["correct_letter"].append(correct_letter)
        results["unknown_letter"].append(unknown_letter)

        # Logps/probs remapped to original labels
        results["logp_option_1"].append(logps_by_letter[idx_o1])
        results["logp_option_2"].append(logps_by_letter[idx_o2])
        results["logp_option_3"].append(logps_by_letter[idx_o3])
        results["logp_option_4"].append(logps_by_letter[idx_o4])
        results["logp_unknown"].append(logps_by_letter[idx_u])

        results["p_option_1"].append(probs_by_letter[idx_o1])
        results["p_option_2"].append(probs_by_letter[idx_o2])
        results["p_option_3"].append(probs_by_letter[idx_o3])
        results["p_option_4"].append(probs_by_letter[idx_o4])
        results["p_unknown"].append(probs_by_letter[idx_u])

        # Per-letter choice probs
        results["p_choice_A"].append(probs_by_letter[0]); results["p_choice_B"].append(probs_by_letter[1])
        results["p_choice_C"].append(probs_by_letter[2]); results["p_choice_D"].append(probs_by_letter[3])
        results["p_choice_E"].append(probs_by_letter[4])

        # Phrase-based probs
        results["p_phrase_A"].append(phrase_probs[0]); results["p_phrase_B"].append(phrase_probs[1])
        results["p_phrase_C"].append(phrase_probs[2]); results["p_phrase_D"].append(phrase_probs[3])
        results["p_phrase_E"].append(phrase_probs[4])

        # Predictions
        results["predicted_label"].append(pred_label)
        results["predicted_text"].append(pred_text)
        results["predicted_letter"].append(pred_letter)
        results["is_correct"].append(bool(is_correct))
        results["predicted_letter_phrase"].append(pred_letter_phrase)

    save_csv_file(results, output_dir=output_dir, filename=filename)
    save_json_file(results, output_dir=output_dir, filename=filename)

# --- NEW: helper to evaluate across all unknown variants ---
def evaluate_all_unknowns(model, tokenizer, dataset, output_dir, filename, language, device, seed):
    unk_list = LANG_SPEC.get(language, LANG_SPEC[DEFAULT_LANG])["UNK"]
    for i, unk in enumerate(unk_list):
        # Ensure reproducibility but vary the shuffle per unknown by offsetting seed
        this_seed = seed + i
        suffixed = f"{filename}_unk{i}"
        print(f"\n=== Running with Unknown #{i}: '{unk}' ===")
        evaluate_model(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            output_dir=output_dir,
            filename=suffixed,
            language=language,
            device=device,
            seed=this_seed,
            unknown_text=unk,
            unknown_id=i,
        )

# --- CHANGED: __main__ to add a flag and route accordingly ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MCQ with shuffled options and letter/phrase probabilities")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory of the saved model (or hub ID)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to MCQ dataset CSV for the given language")
    parser.add_argument("--language", type=str,
                        choices=list(LANG_SPEC.keys()), default="en",
                        help="Language key used in training/eval")
    parser.add_argument("--output_dir", type=str, default="./outputs/",
                        help="Directory to save outputs")
    parser.add_argument("--filename", type=str, default="unlearned_models_mcq_probs",
                        help="Base filename for CSV/JSON outputs (no extension)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device like cuda:0 or cpu")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--all_unknowns", action="store_true",
                        help="Evaluate once per Unknown variant for the selected language")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_dir)
    ds = load_dataset("csv", data_files=args.dataset)["train"]

    # Example subset; adjust as needed
    # ds = ds.select(range(2500, min(2700, len(ds))))
    # ds = ds.shuffle(seed=42).select(range(min(1000, len(ds))))

    if args.all_unknowns:
        evaluate_all_unknowns(
            model=model,
            tokenizer=tokenizer,
            dataset=ds,
            output_dir=args.output_dir,
            filename=args.filename,
            language=args.language,
            device=args.device,
            seed=args.seed,
        )
    else:
        evaluate_model(
            model=model,
            tokenizer=tokenizer,
            dataset=ds,
            output_dir=args.output_dir,
            filename=args.filename,
            language=args.language,
            device=args.device,
            seed=args.seed,
        )
