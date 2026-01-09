import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import math
import argparse
from itertools import islice

def compute_perplexity(model, tokenizer, texts, device, batch_size=16, max_length=512):
    model.eval()
    total_nll = 0.0      # sum of negative log-likelihoods
    total_tokens = 0     # count of non-pad tokens

    for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating Perplexity"):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Mask out padding in labels so pad tokens don't contribute to loss
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        with torch.inference_mode():
            # reduction='mean' over tokens != -100; we convert to total NLL
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss  # mean per-token (over non -100 positions)

        # Count valid tokens in this batch
        valid_tokens = (labels != -100).sum().item()
        total_nll += loss.item() * valid_tokens
        total_tokens += valid_tokens

    # Exact corpus-level perplexity
    avg_nll = total_nll / max(1, total_tokens)
    return math.exp(avg_nll)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    # Use maintained mC4 and updated Hebrew code 'he' (not 'iw')
    languages = ["iw","id","en","fr","ru","ar","ja","fa","hi","ko"]

    results = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # typical for causal LMs

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto",              # ok for single/multi-GPU; or load and .to(device)
        trust_remote_code=True
    )

    for lang in languages:
        print(f"\n--- Evaluating {lang.upper()} ---")

        dataset = load_dataset(
            "allenai/c4",                       # <-- updated dataset id
            lang,
            split="train",
            streaming=True,
            cache_dir="../scratch/",
            trust_remote_code=True
        )

        # sample clean texts
        texts = [x["text"] for x in islice(dataset, args.num_samples * 5) if x["text"] and x["text"].strip()]
        texts = texts[:args.num_samples]

        ppl = compute_perplexity(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length
        )
        results[lang] = ppl
        print(f"Perplexity on mC4 ({lang}): {ppl:.4f}")

    with open(f"{args.filename}.txt", "a") as f:
        f.write(f"\nResults for model: {args.model_path}\n")
        for lang in languages:
            val = results.get(lang)
            f.write(f"{lang}: {val:.4f}\n" if val is not None else f"{lang}: ERROR\n")
        f.write("-" * 40 + "\n")
