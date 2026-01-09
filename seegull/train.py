from pipes import stepkinds

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, default_data_collator, \
    DataCollatorForLanguageModeling
from torch.optim import AdamW
from tqdm.auto import tqdm
from itertools import cycle
from loss_utils import *


# Define optimizer and scheduler
def train_loop(model, pretrained_model, train_dataloader, train_unk_dataloader, normal_dataloader, logger,
               learning_rate=1e-6, unlearn_loss="npo",
               kl_weight=1., unlearn_weight=1., unk_weight=1.):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_epochs = 1
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    LOSS_STEP = 10
    print("kl weight:", kl_weight)
    print("unlearn weight:", unlearn_weight)
    print("unknow weight:", unk_weight)
    # Training loop with gradient ascent
    for epoch in range(num_epochs):
        progress_bar = tqdm(zip(train_dataloader, train_unk_dataloader, cycle(normal_dataloader)),
                            total=len(train_dataloader), desc=f"Epoch {epoch + 1}")
        model.train()

        total_samples = 0
        total_loss = 0
        step_unk_loss = 0
        step_unlearn_loss = 0
        step_loss = 0
        step_samples = 0
        for step, (batch, unk_batch, normal_batch) in enumerate(progress_bar):
            optimizer.zero_grad()
            batch_size = batch['input_ids'].size(0)
            if unlearn_loss == "npo":
                loss = get_npo_loss((batch, unk_batch, normal_batch),model,pretrained_model)
            else:   
                kl_term = 1.
                if kl_weight > 0:
                    kl_term = kl_weight * compute_kl(pretrained_model, model, normal_batch)
                unlearn_term = unlearn_weight * get_answer_loss('ga', batch, model)
                unknown_term = unk_weight * get_answer_loss('gd', unk_batch, model)
                loss = (unlearn_term + unknown_term + kl_term)


            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            if unlearn_loss !="npo":
                step_unk_loss += unknown_term.item() * batch_size
                step_unlearn_loss += unlearn_term.item() * batch_size
            step_loss += loss.item() * batch_size
            step_samples += batch_size

            # Display step loss every 100 steps
            if (step + 1) % LOSS_STEP == 0:
                average_step_loss = step_loss / step_samples
                if unlearn_loss !="npo":
                    average_step_unk_loss = step_unk_loss / step_samples
                    average_step_unlearn_loss = step_unlearn_loss / step_samples
                    logger.info(f"Average unk loss after step {step + 1} of epoch {epoch + 1}: {average_step_unk_loss}")
                    logger.info(
                    f"Average unlearn loss after step {step + 1} of epoch {epoch + 1}: {average_step_unlearn_loss}")
                
                logger.info(f"Average loss after step {step + 1} of epoch {epoch + 1}: {average_step_loss}")
                step_loss = 0
                step_unlearn_loss = 0
                step_unk_loss = 0
                step_samples = 0

            progress_bar.set_postfix({"loss": loss.item()})
        average_loss = total_loss / total_samples
        logger.info(f"Average loss after epoch {epoch + 1}: {average_loss}")

    logger.info("Fine-tuning completed.")
    return model
