import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import math
import argparse


def compute_perplexity(model_path, dataset, batch_size=16, max_length=512, device="cuda"):
    """
    Computes perplexity of a causal language model on a given dataset.
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path,  torch_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()

    losses = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating Perplexity"):
        batch = dataset[i:i + batch_size]
        encodings = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)
    return perplexity

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    args = parser.parse_args()
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")["test"]
    texts = [x["text"] for x in dataset if x["text"].strip()]

    # Path to your Hugging Face model (can be local or model hub)
    # model_path = "CohereForAI/aya-expanse-8b"
    # model_path = "../scratch/tofu_finetuned_5epoch_aya_all_lang_5e5/" 
    # model_path = "../scratch/tofu_finetuned_5epoch_aya_all_lang_5e5/grad_diff_2e-05_forget01_5_fr/" 
    model_path = args.model_path
    # Compute perplexity
    ppl = compute_perplexity(model_path, texts)
    print(f"Perplexity: {ppl:.4f}")








