import torch
from transformers import DataCollatorForLanguageModeling

# -----------------------------
# Language specifications (10 langs)
# -----------------------------
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
           "UNK": ["לא ניתן לקבוע", "לא ידוע", "אין מספיק מידע", "לא ברור"]},  # Hebrew
    "id": {"Q": "Pertanyaan: ", "A": "Jawaban: ", "Q_END": " \n",
           "UNK": ["Tidak dapat ditentukan", "Tidak diketahui", "Informasi tidak cukup", "Tidak jelas"]},
    "ja": {"Q": "質問: ", "A": "回答: ", "Q_END": " \n",
           "UNK": ["判断不能", "不明", "情報が不十分", "不明確"]},
    "ko": {"Q": "질문: ", "A": "답변: ", "Q_END": " \n",
           "UNK": ["판단할 수 없음", "알 수 없음", "정보가 충분하지 않음", "불분명"]},
    "ru": {"Q": "Вопрос: ", "A": "Ответ: ", "Q_END": " \n",
           "UNK": ["Невозможно определить", "Неизвестно", "Недостаточно информации", "Неясно"]},
}

# Fallback language if an unsupported code is passed
DEFAULT_LANG = "en"


def _lang(key: str, language: str) -> str:
    """Fetch a language-specific tag (Q, A, Q_END) with fallback."""
    spec = LANG_SPEC.get(language, LANG_SPEC[DEFAULT_LANG])
    return spec[key]


def _random_unknown(language: str) -> str:
    # import random
    # spec = LANG_SPEC.get(language, LANG_SPEC[DEFAULT_LANG])
    # return random.choice(spec["UNK"])

    return LANG_SPEC.get(language, LANG_SPEC[DEFAULT_LANG])["UNK"][1]

def _build_qa_strings(q: str, a: str, language: str, eos_token: str):
    """Return (full_text, prefix_for_start_loc) where start_locs = len(tokenize(prefix))-1."""
    Q, A, Q_END = _lang("Q", language), _lang("A", language), _lang("Q_END", language)
    # Full training text
    text = f"{Q}{q}{Q_END}{A}{a}{eos_token}"
    # Prefix up to (but not including) the answer tokens
    prefix = f"{Q}{q}{Q_END}{A}"
    return text, prefix


def _tokenize_and_pack(tokenizer, full_text: str, prefix_text: str):
    """Tokenize and compute start_locs from the prefix length."""
    tok_full = tokenizer(full_text, truncation=True, padding="longest")
    tok_prefix = tokenizer(prefix_text, truncation=False, padding=False)
    start_loc = len(tok_prefix["input_ids"]) - 1
    return tok_full["input_ids"], tok_full["attention_mask"], start_loc




def get_seegull_dataloaders(tokenizer, dataset, seegull_ds, language="en"):
    """
    - seegull_ds: MCQ dataset with columns:
        MCQ_Question, Correct_Option, and option columns (e.g., A,B,C,D or matching keys)
    - dataset: TruthfulQA-like dataset with columns: 'question', 'best_answer'
    """
    EOS = tokenizer.eos_token or ""

    # MCQ: correct option
    tokenized_dataset = seegull_ds.map(
        preprocess_mcq,
        batched=True,
        batch_size=100,
        remove_columns=seegull_ds.column_names,
        fn_kwargs={"tokenizer": tokenizer, "EOS": EOS, "language": language},
    )
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "start_locs"])

    # MCQ: unknown-option augmentation
    unk_dataset = seegull_ds.map(
        preprocess_mcq_unk,
        batched=True,
        batch_size=100,
        remove_columns=seegull_ds.column_names,
        fn_kwargs={"tokenizer": tokenizer, "EOS": EOS, "language": language},
    )
    unk_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "start_locs"])

    # TruthfulQA-style normalization
    normalized_dataset = dataset.map(
        preproccess_truthfulqa,
        batched=True,
        batch_size=100,
        remove_columns=dataset.column_names,
        fn_kwargs={"tokenizer": tokenizer, "EOS": EOS, "language": language},
    )
    normalized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "start_locs"])

    # Collator & DataLoaders
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=6, collate_fn=data_collator, shuffle=True)
    train_unk_dataloader = torch.utils.data.DataLoader(unk_dataset, batch_size=6, collate_fn=data_collator, shuffle=True)
    normal_dataloader = torch.utils.data.DataLoader(normalized_dataset, batch_size=6, collate_fn=data_collator, shuffle=True)

    return train_dataloader, train_unk_dataloader, normal_dataloader


def preproccess_truthfulqa(examples, tokenizer, EOS, language="en"):
    """
    Input: Dict[List] with keys: 'question', 'best_answer'
    Output: Dict[List] -> 'input_ids', 'attention_mask', 'start_locs'
    """
    results = {"input_ids": [], "attention_mask": [], "start_locs": []}
    for q, a in zip(examples["question"], examples["best_answer"]):
        full_text, prefix = _build_qa_strings(q, a, language, EOS)
        input_ids, attn, start_loc = _tokenize_and_pack(tokenizer, full_text, prefix)
        results["input_ids"].append(input_ids)
        results["attention_mask"].append(attn)
        results["start_locs"].append(start_loc)
    return results


def preprocess_mcq(examples, tokenizer, EOS, language="en"):
    """
    Input: Dict[List], required keys:
      - 'MCQ_Question'
      - 'Correct_Option' (string key pointing to the correct option column)
      - option columns that match 'Correct_Option' values
    """
    results = {"input_ids": [], "attention_mask": [], "start_locs": []}
    for row in zip(*examples.values()):
        row_data = dict(zip(examples.keys(), row))
        q = row_data["MCQ_Question"]
        correct_option_key = row_data["Correct_Option"]
        a = row_data[correct_option_key]

        full_text, prefix = _build_qa_strings(q, a, language, EOS)
        input_ids, attn, start_loc = _tokenize_and_pack(tokenizer, full_text, prefix)
        results["input_ids"].append(input_ids)
        results["attention_mask"].append(attn)
        results["start_locs"].append(start_loc)
    return results


def preprocess_mcq_unk(examples, tokenizer, EOS, language="en"):
    """
    Same as preprocess_mcq, but injects a language-appropriate 'Unknown' answer.
    """
    results = {"input_ids": [], "attention_mask": [], "start_locs": []}
    for row in zip(*examples.values()):
        row_data = dict(zip(examples.keys(), row))
        q = row_data["MCQ_Question"]
        a = _random_unknown(language)

        full_text, prefix = _build_qa_strings(q, a, language, EOS)
        input_ids, attn, start_loc = _tokenize_and_pack(tokenizer, full_text, prefix)
        results["input_ids"].append(input_ids)
        results["attention_mask"].append(attn)
        results["start_locs"].append(start_loc)
    return results
