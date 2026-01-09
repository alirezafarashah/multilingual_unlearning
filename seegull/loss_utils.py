import torch
from torch import nn

import torch.nn.functional as F


def compute_batch_nll(model, inputs):
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    labels = inputs["labels"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    logits = outputs.logits

    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)

    return loss, outputs



def compute_dpo_loss(model, ref_model, win_inputs=None, lose_inputs=None, beta=1.0):
    if win_inputs is None and lose_inputs is None:
        raise ValueError("Both win_inputs and lose_inputs can't be None")

    device = next(model.parameters()).device
    win_log_ratio, lose_log_ratio = 0.0, 0.0
    win_outputs, lose_outputs = None, None

    if win_inputs is not None:
        win_loss, win_outputs = compute_batch_nll(model, win_inputs)
        with torch.no_grad():
            win_ref_loss, _ = compute_batch_nll(ref_model, win_inputs)
            win_ref_loss = win_ref_loss.to(device)
        win_log_ratio = -(win_loss - win_ref_loss)

    if lose_inputs is not None:
        lose_loss, lose_outputs = compute_batch_nll(model, lose_inputs)
        with torch.no_grad():
            lose_ref_loss, _ = compute_batch_nll(ref_model, lose_inputs)
            lose_ref_loss = lose_ref_loss.to(device)
        lose_log_ratio = -(lose_loss - lose_ref_loss)

    loss = -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
    return loss, (win_outputs, lose_outputs)

    
def compute_retain_loss(model, retain_inputs):
    retain_input_ids = retain_inputs["input_ids"]
    retain_labels = retain_inputs["labels"]
    retain_attention_mask = retain_inputs["attention_mask"]
    # retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
    retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
    retain_loss = 0.0
    retain_loss += retain_outputs.loss
    return retain_loss



def get_npo_loss(inputs, model, ref_model, beta=1., gamma=1., alpha=1.):      
    forget_inputs, retain_inputs, normal_inputs = inputs
    forget_loss, forget_outputs = compute_dpo_loss(
        model=model,
        ref_model=ref_model,
        win_inputs=None,
        lose_inputs=forget_inputs,
        beta=beta,
    )
    
    retain_loss = compute_kl(ref_model, model, normal_inputs)
    retain_loss_2 = compute_retain_loss(model=model, retain_inputs=retain_inputs.to(model.device))

    loss = gamma * forget_loss + alpha * (retain_loss + retain_loss_2)
    return loss

def get_answer_loss(operation, batch, model):
    #   operation: either "ga" (gradient ascent) or "gd" (gradient descent).

    device = model.device
    input_ids, attention_mask, start_locs, labels = (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["start_locs"],
        batch["labels"].to(device),
    )
    outputs = model(input_ids, attention_mask=attention_mask)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # Shift one to predict next token.
    # Logits shape is (batch size, sequence length, vocab size)
    shift_logits = outputs.logits[:, :-1, :]
    # Label shape is (batch size, sequence length)
    shift_labels = labels[:, 1:]
    losses = []
    for bid in range(input_ids.shape[0]):
        one_inp, one_st = input_ids[bid], start_locs[bid]

        # GA or GD.
        # output has dimension of vocabulary size and label indicate the index of true token
        position_loss = loss_fct(shift_logits[bid], shift_labels[bid])
        if operation == "ga":
            position_loss = -position_loss

        position_weight = torch.zeros_like(one_inp)
        assert len(position_weight) == len(position_loss) + 1
        position_weight[one_st:] = 1  # only consider answer part

        # Ignore the padding part.
        position_weight[one_inp == 1] = 0
        if position_weight.sum() > 0:
            position_weight = position_weight / position_weight.sum()

        one_loss = (position_weight[:-1] * position_loss).sum()
        losses.append(one_loss)
    final_loss = torch.stack(losses).mean()
    return final_loss


def compute_kl(pretrained_model, current_model, batch):
    device_0 = current_model.device
    normal_outputs = current_model(
        batch["input_ids"].to(device_0),
        attention_mask=batch["attention_mask"].to(device_0),
        labels=batch["labels"].to(device_0),
    )

    device_1 = pretrained_model.device
    with torch.no_grad():
        pretrained_outputs = pretrained_model(
            batch["input_ids"].to(device_1),
            attention_mask=batch["attention_mask"].to(device_1),
            labels=batch["labels"].to(device_1),
        )

    # P: pretrained model; Q: current model.
    prob_p = torch.nn.functional.softmax(pretrained_outputs.logits, -1).to(device_0)
    prob_q = torch.nn.functional.softmax(normal_outputs.logits, -1).to(device_0)

    loss = -(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()

    return loss
