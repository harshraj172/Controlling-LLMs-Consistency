
import os
import pdb
import sys
import glob
import numpy as np
import pickle as pkl
import torch
import random
import json
import re
import shutil

from collections import defaultdict, Counter
from tqdm import tqdm as progress_bar
from copy import deepcopy
from transformers import get_scheduler
from torch.optim import AdamW
import torch_optimizer as ada_optim
from arguments import solicit_params
from data import get_dataloader, process_data
from evalute import run_eval
from load import load_best_model, load_model, load_tokenizer

from models import CausalEmbedding, Seq2SeqEmbedding


def setup_optimization(args, model, total_steps):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    if args.task == 'soft_prompt':
        # model is actually soft prompt embeds
        optimizer_grouped_parameters = model.parameters()
    else:
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": args.weight_decay,
             },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0},
        ]

    warmup = int(total_steps * args.warmup_steps)
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, weight_decay=args.weight_decay)

    schedule_type = 'cosine' if torch.cuda.is_available() else 'constant'
    scheduler = get_scheduler(
        schedule_type, optimizer, num_warmup_steps=warmup, num_training_steps=total_steps)
    return optimizer, scheduler


def run_train_loop(args, model, datasets, soft_embeds=None):
    dataset, dev_dataset = datasets['train'], datasets['dev']
    train_dataloader = get_dataloader(args, dataset)
    total_steps = len(
        train_dataloader) // args.grad_accum_steps * args.n_epochs

    if soft_embeds:
        optimizer, scheduler = setup_optimization(
            args, soft_embeds, total_steps)
    else:
        optimizer, scheduler = setup_optimization(args, model, total_steps)

    best_score = 0
    for epoch_count in range(args.num_epochs):
        model.train()

        losses = []
        for step, batch in enumerate(train_dataloader):
            inputs, targets = dataset.collate(args, batch)
            outputs = model(**inputs, labels=targets)
            loss = outputs.loss / args.grad_accum_steps
            losses.append(outputs.loss.item().numpy())
            loss.backward()

        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        mean_loss = np.mean(losses)
        eval_res = run_eval(args, model, dev_dataset)
        print(f"""
        epoch: {epoch_count}
        mean loss: {mean_loss}
        results: {eval_res}
        """)
        if eval_res[args.metric] >= best_score:
            best_score = eval_res
            if soft_embeds:
                save_soft_prompt(args, soft_embeds)
            else:
                save_model(model)

    return model


def run_prompt_train(args, model, datasets):
    # freeze the large LM
    parameters = list(model.parameters())
    # can also tune the vocab embeddings by freezing first params
    # for param in parameters:
    for param in parameters:
        param.requires_grad = False

    # create and then set the soft prompt embeddings
    if args.model == 'gpt':
        soft_prompt_embed = CausalEmbedding(
            model.get_input_embeddings(), args.n_tokens)
    else:
        soft_prompt_embed = Seq2SeqEmbedding(
            model.get_input_embeddings(), args.n_tokens)
    model.set_input_embeddings(soft_prompt_embed)

    model = run_train_loop(args, model, datasets, soft_prompt_embed)
    return model


if __name__ == "__main__":
    args = solicit_params()
    tokenizer = load_tokenizer(args)
    datasets = process_data(args, tokenizer)
    model = load_model(args)
    if args.do_train:
        if args.task == 'soft_prompt':
            run_prompt_train(args, model, datasets)
        elif args.task in ['fine_tune']:
            run_train_loop(args, model, datasets)
    else:
        model = load_best_model(args, tokenizer)
        run_eval(args, model, datasets['test'], 'test')
