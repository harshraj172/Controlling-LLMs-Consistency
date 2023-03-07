import os, pdb, sys
import json
import re
import random
import glob
import csv
import pickle as pkl
import numpy as np
import pandas as pd
import torch
import errno

from tqdm import tqdm as progress_bar
from components.soft_embedder import AttributeEmbedding, CausalEmbedding, Seq2SeqEmbedding
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, \
                         T5ForConditionalGeneration, T5Config, T5Tokenizer, logging
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, \
                        GPTNeoXForCausalLM, GPTNeoXConfig, GPTNeoXTokenizerFast, \
                        GPTJForCausalLM, GPTJConfig, AutoModelForCausalLM

from assets.static_vars import device, dtype, DATASETS, CHECKPOINTS, STOP_TOKENS, CROSS
from components.models import SentenceBERT, CVAEModel, SingleClassifier
from static import MODEL_DIR, MODELS
from utils.help import model_match, find_best_model_path


def load_tokenizer(args):
  token_ckpt = MODELS[args.model][args.size]
  if args.model == 't5':
    tokenizer = T5Tokenizer.from_pretrained(token_ckpt)
  elif args.model == 'gpt':
    tokenizer = AutoTokenizer.from_pretrained(token_ckpt)
  return tokenizer

def load_model(args, ckpt_path=''):
  ckpt_name = MODELS[args.model][args.size] if len(ckpt_path) == 0 else ckpt_path

  if args.model == 'gpt':
    if args.size == 'giant':
      # https://huggingface.co/docs/transformers/model_doc/gptj
      model = GPTJForCausalLM.from_pretrained(ckpt_name, low_cpu_mem_usage=True)
    else:
      config = GPT2Config.from_pretrained(ckpt_name)
      model = GPT2LMHeadModel.from_pretrained(ckpt_name, config=config, low_cpu_mem_usage=True)
  elif args.model == 't5':
    model = T5ForConditionalGeneration.from_pretrained(ckpt_name)

  if args.n_gpu > 1:
    model.parallelize()
  else:
    model.to(device)
  return model

def load_best_model(args, tokenizer):
  load_dir = MODEL_DIR
  print(f'Loading best finetuned model from {load_dir} ...')

  if len(args.checkpoint) > 0:
    top_filename = args.checkpoint
    top_folder = os.path.join(load_dir, top_filename)
  else:
    # folders = glob.glob(load_dir)
    top_folder = find_best_model_path(load_dir, metric=args.metric)
  if top_folder is None:
    # raise RuntimeError(f'No models were found in {load_dir}')
    print(f'No checkpoints were found in {load_dir}, loading the default parameters')
    ckpt_path = ''
  else:
    ckpt_path = top_folder
    print(f'Attempting to load {ckpt_path} as best model')
  # checkpoint = torch.load(ckpt_path, map_location='cpu')
  # model.load_state_dict(checkpoint)
  model = load_model(args, tokenizer, load_dir, ckpt_path)
  return model

def load_best_attribute_embedder(args, model, exp_logger):
  load_dir = exp_logger.save_path
  print(f'Loading best prompt from {load_dir} ...')

  if len(args.checkpoint) > 0 and not args.accelerate:
    top_filename = args.checkpoint
    top_folder = os.path.join(load_dir, top_filename)
  else:
    top_folder = find_best_model_path(load_dir, metric=args.metric)

  if top_folder is None:
    raise RuntimeError(f'No checkpoints were found in {load_dir}')

  ckpt_path = top_folder.replace('attr_map_', '').replace('attention_', '')
  print(f'Attempting to load {ckpt_path} as best model')
  original_emb = model.get_input_embeddings()
  attr_embedding = AttributeEmbedding.from_saved_embedding(args, original_emb, ckpt_path)
  if args.model == 'gpt':
    attr_embedding.instruction_prompt = CausalEmbedding.from_saved_embedding(
                                                           args, original_emb, ckpt_path)
  else:
    attr_embedding.instruction_prompt = Seq2SeqEmbedding.from_saved_embedding(
                                                           args, original_emb, ckpt_path)
  attr_embedding.to(device)
  attr_embedding.instruction_prompt.to(device)
  model.set_input_embeddings(attr_embedding)
  # ensure embeddings are all on the same device now
  # if the model isn't parallelized
  if args.n_gpu <= 1:
    model.to(device)
  return model, attr_embedding


def load_best_soft_prompt(args, model, exp_logger):
  load_dir = exp_logger.save_path

  if len(args.checkpoint) > 0:
    top_filename = args.checkpoint
    top_folder = os.path.join(load_dir, top_filename)
  else:
    top_folder = find_best_model_path(load_dir, args.metric)

  if top_folder is None:
    raise RuntimeError(f'No checkpoints were found in {load_dir}')

  ckpt_file = top_folder
  print(f'Attempting to load {ckpt_file} as best model')

  if args.model == 'gpt':
    soft_prompt_embed = CausalEmbedding.from_saved_embedding(args, model.get_input_embeddings(), ckpt_file)
  else:
    soft_prompt_embed = Seq2SeqEmbedding.from_saved_embedding(args, model.get_input_embeddings(), ckpt_file)
  soft_prompt_embed.to(device)
  model.set_input_embeddings(soft_prompt_embed)
  # ensure embeddings are all on the same device now
  # if the model isn't parallelized
  if args.n_gpu <= 1:
    model.to(device)
  return model

