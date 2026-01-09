import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, default_data_collator, \
    DataCollatorForLanguageModeling
from datasets import load_dataset

import copy
import argparse
import logging
import os
from pathlib import Path

from data_utils import get_seegull_dataloaders

from train import train_loop
from eval import evaluate_model

import random
import numpy as np

from huggingface_hub import login



def load_pretrained_model(model):
    pretrained_model = copy.deepcopy(model)
    pretrained_model.eval()
    return pretrained_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def save_model(model, tokenizer, output_dir):
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

if __name__ == '__main__':
    home_dir = Path.home()
    parser = argparse.ArgumentParser(description="Unlearn biases in large language models.")
    parser.add_argument("--model_name", type=str, default="CohereForAI/aya-expanse-8b",
                        help="The name of the pre-trained model.")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate for training.")
    parser.add_argument("--weights", type=float, nargs=3, default=[1.0, 0.25, 0.5],
                        help="Weights for kl_weight, unlearn_weight, and unk_weight (in that order).")
    parser.add_argument("--output_dir", type=str, default=home_dir / "scratch/models/",
                        help="The output directory.")
    parser.add_argument("--dataset", type=str, default="./mcq_stereotype_dataset.csv",
                        help="The output directory.")
    parser.add_argument("--log_dir", type=str, default="./log", help="Log dir to capture the output.")
    parser.add_argument("--language", type=str, choices=["en", "hi", "fr", "ar", "fa", "iw", "id", "ko", "ja", "ru"], default="en",
                        help="Language for unlearning")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--normal_data", type=str, default="truthful_qa")
    parser.add_argument("--unlearn_loss", type=str, default="grad_diff_KL")
    args = parser.parse_args()
    set_seed(args.seed)

    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, "training.log")
    logger = setup_logger(log_file)
    logger.info("Parsed Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")


    if args.normal_data == "truthful_qa":
        normal_dataset = load_dataset("truthful_qa", 'generation', split="validation")
    else:
        normal_dataset = None
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    print("model loaded")

    seegull_ds = load_dataset("csv", data_files=args.dataset)
    seegull_ds = seegull_ds['train']
    train_dataloader, train_unk_dataloader, normal_dataloader = get_seegull_dataloaders(tokenizer, normal_dataset,
                                                                                        seegull_ds,
                                                                                        args.language)

    # Convert weights from string to float
    kl_weight, unlearn_weight, unk_weight = args.weights

    # Move models on GPU
    model.to("cuda:0")
    if kl_weight != 0:
        pretrained_model = load_pretrained_model(model)
        print("pretrained model loaded")
        pretrained_model.to("cuda:1")
    else:
        pretrained_model = None
    print("model loaded on GPU")

    logger.info("Starting training loop")
    model = train_loop(
        model, pretrained_model, train_dataloader, train_unk_dataloader, normal_dataloader, logger,
        learning_rate=args.learning_rate, unlearn_loss=args.unlearn_loss, kl_weight=kl_weight, unlearn_weight=unlearn_weight, unk_weight=unk_weight
    )
    
    save_model(model, tokenizer,args.output_dir)
    print("saved successfully")
 
 
