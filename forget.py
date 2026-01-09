from data_module import TextForgetDatasetQA
from dataloader import CustomTrainerForgetting, custom_data_collator_forget, custom_data_collator_forget_kl
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed

import hydra 
import transformers
import os
from pathlib import Path
from utils import get_model_identifiers_from_yaml


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

@hydra.main(version_base=None, config_path="config", config_name="forget")
def main(cfg):
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")


    set_seed(cfg.seed)

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    if cfg.model_path is None:
        cfg.model_path = model_cfg["ft_model_path"]

    print("######################")
    print("Saving to: ", cfg.save_dir)
    print("######################")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    max_length = 500

    torch_format_dataset = TextForgetDatasetQA(cfg.data_path, tokenizer=tokenizer, model_family = cfg.model_family, max_length=max_length, split=cfg.split, loss_type=cfg.forget_loss, language=cfg.language)
    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    steps_per_epoch = len(torch_format_dataset)//(batch_size*gradient_accumulation_steps*num_devices)

    max_steps = int(cfg.num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps)
    print(f"max_steps: {max_steps}")
    print("batch_size:", batch_size)
    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size//torch.cuda.device_count(),
            per_device_eval_batch_size=batch_size//torch.cuda.device_count(),
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=max(1, steps_per_epoch),
            max_steps=max_steps,
            learning_rate=cfg.lr,
            bf16=True,
            bf16_full_eval=True,
            logging_steps=max(1,max_steps//20),
            logging_dir=f'{cfg.save_dir}/logs',
            output_dir=cfg.save_dir,
            optim="paged_adamw_32bit",
            save_strategy="steps" if cfg.save_model and (not cfg.eval_only) else "no",
            save_steps=steps_per_epoch,
            save_only_model=True,
            ddp_find_unused_parameters= False,
            # deepspeed='config/ds_config.json',
            weight_decay = cfg.weight_decay,
            eval_steps = steps_per_epoch,
            eval_strategy = "steps" if cfg.eval_while_train else "no",
            seed=cfg.seed

        )
    # Load the main model on GPU 0 (cuda:0)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, 
        attn_implementation="flash_attention_2" if model_cfg["flash_attention2"] == "true" else None,
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    ).to("cuda:0")  
    
    # Load the oracle model on GPU 1 (cuda:1), only if needed
    oracle_model = None
    if cfg.forget_loss == "KL" or cfg.forget_loss == "grad_diff_KL" or cfg.forget_loss == "npo":
        oracle_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path, 
            attn_implementation="flash_attention_2" if model_cfg["flash_attention2"] == "true" else None,
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        ).to("cuda:1") 
        oracle_model.eval()

    model.generation_config.do_sample = True
    
    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()


    data_collator = custom_data_collator_forget
    if cfg.forget_loss == "grad_diff_KL":
        data_collator = custom_data_collator_forget_kl
       
    trainer = CustomTrainerForgetting(
        model=model,
        tokenizer=tokenizer,
        train_dataset=torch_format_dataset,
        eval_dataset = torch_format_dataset,
        compute_metrics=None,                # the callback for computing metrics, None in this case since you're doing it in your callback
        # callbacks=[GlobalStepDeletionCallback],
        args=training_args,
        data_collator=data_collator,
        oracle_model = oracle_model,
        beta = cfg.beta,
        gamma= cfg.gamma,
        forget_loss = cfg.forget_loss,
        eval_cfg = cfg.eval,
    )
    model.config.use_cache = False  
    # trainer.train()
    if cfg.eval_only:
        trainer.evaluate()
    else:
        trainer.train()

    #save the tokenizer
    if cfg.save_model and (not cfg.eval_only):
        model.save_pretrained(cfg.save_dir)
        tokenizer.save_pretrained(cfg.save_dir)

    #delete all "global_step*" files in the save_dir/checkpoint-*/ directories
    if local_rank == 0:
        for file in Path(cfg.save_dir).glob("checkpoint-*"):
            for global_step_dir in file.glob("global_step*"):
                #delete the directory
                import shutil
                shutil.rmtree(global_step_dir)



if __name__ == "__main__":
    main()

