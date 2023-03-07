import os, pdb, sys
import random
import json
import math
import pickle as pkl
import numpy as np
import re
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from static import DATA_DIR, device

class BaseDataset(Dataset):
  def __init__(self, args, examples, tokenizer, split):
    self.args = args
    self.data = examples
    self.tokenizer = tokenizer
    self.name = 'base-dataset'
    self.split = split
    self.shuffle = (split == 'train')

    self.dataset = args.dataset
    self.model_type = args.model

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]

  def _pad_null(self, targets, max_tokens, direction='right'):
    """ null means value of -100 which tells the model to ignore when computing the loss.
    The additional benefit is being able to specify the direction of padding. """
    padded = []
    for vector in targets.input_ids:
      tensor = torch.tensor(vector[:max_tokens])   # truncate to max sequence length
      diff = max_tokens - tensor.shape[0]
      if diff > 0:
        if direction == 'left':
          tensor = nn.functional.pad(tensor, (diff, 0), value=-100)
        elif direction == 'right':
          tensor = nn.functional.pad(tensor, (0, diff), value=-100)
      padded.append(tensor)

    target = torch.stack(padded).to(device)
    return target

  @staticmethod
  def target_to_tensor(targets, max_len):
    """ Transform a list of strings into a list of tensors of the same length """
    tensors = []
    for label_string in targets:
      numbered = [ord(char) for char in label_string]

      if len(numbered) > max_len:
        tensors.append(numbered[:max_len])
      elif len(numbered) < max_len:
        gap = max_len - len(numbered)
        filler = [ord('$')] * gap
        tensors.append(numbered + filler)
      else:
        tensors.append(numbered)

    transformed = torch.tensor(tensors, dtype=torch.long).to(device)
    return transformed

  @staticmethod
  def tensor_to_target(tensors):
    """ Transform the list of label tensors back into the original strings """
    transformed = []
    for label_tensor in tensors.tolist():
      lettered = [chr(char) for char in label_tensor]

      ptr = len(lettered) - 1
      while lettered[ptr] == '$':
        ptr -= 1
      string = ''.join(lettered[:ptr+1])

      transformed.append(string)
    return transformed

  def collate_lm(self, args, examples):
    raise NotImplementedError

  def collate_seq2seq(self, args, examples):
    raise NotImplementedError

  def collate(self, args, examples):
    if self.model_type in ['gpt']:
      return self.collate_lm(args, examples)
    elif self.model_type in ['t5']:
      return self.collate_seq2seq(args, examples)

  def collate_func(self, examples):
    return examples
  
class ConsistencyDataset(BaseDataset):
  def __init__(self, args, examples, tokenizer, split):
    super().__init__(args, examples, tokenizer, split)
    self.name = 'consistency-dataset'

  def collate_seq2seq(self, args, examples):
    """transforms a batch of examples into a features dict that can be fed into a T5 or BART model"""
    inputs, targets = [], []
    # input,output,input pp,output pp,pp generation method,consistency score
    for example in examples:
      inputs.append(example['input'])
      targets.append(example['output'])

    inputs = self.tokenizer(inputs, padding='longest', max_length=args.source_max_len,
                          truncation=True, pad_to_multiple_of=8, return_tensors='pt').to(device)

    if args.task == 'soft_prompt':
      batch_size = len(examples)      # set to negative to differentiate from decoder inputs
      prompt_tokens = torch.full((batch_size,args.n_tokens), -1, device=device)
      inputs['input_ids'] = torch.cat([prompt_tokens, inputs['input_ids']], 1)
      prompt_mask = torch.ones((batch_size, args.n_tokens)).to(device)
      inputs['attention_mask'] = torch.cat([prompt_mask, inputs['attention_mask']], 1)

    if self.split == 'train':
      targets = self.tokenizer(targets)  # we do not want to return tensors
      max_vector_len = min(max([len(v) for v in targets.input_ids]), args.target_max_len)
      target_tensor = self._pad_null(targets, max_vector_len, direction='right')
      return inputs, target_tensor

    else:
      return inputs, targets

  def collate_lm(self, args, examples):
    """transforms a batch of examples into a features dict that can be fed into a GPT-like model"""
    inputs, targets = [], []
    eos = self.tokenizer.eos_token

    for example in examples:
      target = example['output']

      input = f"Q: {example['input']}\nA: "
      max_length = args.source_max_len
      if self.split == 'train':
        input +=  f"{target}{eos}"
        max_length = args.source_max_len + args.target_max_len
        
      inputs.append(input)
      targets.append(target)

    inputs = self.tokenizer(inputs, padding=True, max_length=max_length,
                              truncation=True, return_tensors='pt').to(device)

    if args.task == 'soft_prompt':
      batch_size = len(examples)
      prompt_tokens = torch.full((batch_size, args.n_tokens), 1, device=device)
      inputs['input_ids'] = torch.cat([prompt_tokens, inputs['input_ids']], 1)
      prompt_attn = torch.full((batch_size, args.n_tokens), 1, device=device)
      inputs['attention_mask'] = torch.cat([prompt_attn, inputs['attention_mask']], 1)

    if self.split == 'train':
      return inputs, inputs['input_ids']
    else:
      return inputs, targets
    

def get_dataloader(args, dataset, split='train'):
  sampler = RandomSampler(dataset) if dataset.shuffle else SequentialSampler(dataset)
  collate = dataset.collate_func
  dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=collate)
  print(f"Loaded {split} data with {len(dataloader)} batches")
  return dataloader

# we want to construct a dataset to ensure
# samples are maximally consistent
def construct_samples(examples):
  samples = []
  for example in examples:
      samples.append({
        'input': example['input'],
        'output': example['output']
      })
      samples.append({
        'input': example['input'],
        'output': example['output pp']
      })
      samples.append({
        'input': example['input pp'],
        'output': example['output']
      })
      samples.append({
        'input': example['input pp'],
        'output': example['output pp']
      })

  return samples



def process_data(args, tokenizer):
  data_path = DATA_DIR
  if args.dataset == 'consistent':
    data_path = os.path.join(DATA_DIR, 'consistent')
  elif args.dataset == 'inconsistent':
    data_path = os.path.join(DATA_DIR, 'inconsistent')
  
  datasets = {}
  for split in ['train', 'test', 'dev']:
    original_df = pd.read_csv(os.path.join(data_path, f'{split}.csv'))
    samples = construct_samples(original_df)
    datasets[split] = ConsistencyDataset(args, samples, tokenizer, split)
  
  return datasets
