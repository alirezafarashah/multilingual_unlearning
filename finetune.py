from data_module import TextDatasetQA, custom_data_collator
from dataloader import CustomTrainer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
from transformers.trainer_utils import get_last_checkpoint
from hydra.utils import to_absolute_path

import hydra 
import transformers
import os
from pathlib import Path
from omegaconf import OmegaConf
from utils import get_model_identifiers_from_yaml

@hydra.main(version_base=None, config_path="config", config_name="finetune")
def main(cfg):
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
    set_seed(cfg.seed)
    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    # save the cfg file

    if os.environ.get('LOCAL_RANK') is None or local_rank == 0:
        with open(f'{cfg.save_dir}/cfg.yaml', 'w') as f:
            OmegaConf.save(cfg, f)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    max_length = 500
    torch_format_dataset = TextDatasetQA(cfg.data_path, tokenizer=tokenizer, model_family = cfg.model_family, max_length=max_length)

    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    # --nproc_per_node gives the number of GPUs per = num_devices. take it from torchrun/os.environ
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    
    max_steps = int(cfg.num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps*num_devices)
    # max_steps=5
    print(f"max_steps: {max_steps}")
    save_dir = to_absolute_path(cfg.save_dir)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=max(1, max_steps//cfg.num_epochs),
            max_steps=max_steps,
            learning_rate=cfg.lr,
            bf16=True,
            bf16_full_eval=True,
            logging_steps=max(1,max_steps//20),
            logging_dir=f'{cfg.save_dir}/logs',
            output_dir=save_dir,
            optim="paged_adamw_32bit",
            save_steps=max_steps,
            save_strategy="epoch",
            save_only_model=False,
            ddp_find_unused_parameters= False,
            eval_strategy="no",
           # deepspeed='config/ds_config.json',
            weight_decay = cfg.weight_decay,
            seed = cfg.seed,
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="flash_attention_2" if model_cfg["flash_attention2"] == "true" else None,
                                                 torch_dtype=torch.bfloat16, trust_remote_code = True,)

    trainer = CustomTrainer(
        model=model,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        args=training_args,
        data_collator=custom_data_collator,
    )
    model.config.use_cache = False

    last_ckpt = None
    if os.path.isdir(save_dir):
        last_ckpt = get_last_checkpoint(save_dir)   
    if last_ckpt:
        print(f"Resuming from checkpoint: {last_ckpt}")
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        trainer.train()


    model.save_pretrained(cfg.save_dir)
    tokenizer.save_pretrained(cfg.save_dir)

if __name__ == "__main__":
    main()
