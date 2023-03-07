
import os, pdb, sys, glob
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
from data import get_dataloader
import evaluate

from models import CausalEmbedding, Seq2SeqEmbedding

rouge = evaluate.load('rouge')

class PP_Detector():
    def __init__(self, tok_path="domenicrosati/deberta-v3-large-finetuned-paws-paraphrase-detector", \
                 model_path="domenicrosati/deberta-v3-large-finetuned-paws-paraphrase-detector", max_len=30):
        super(PP_Detector, self).__init__()
        self.detection_tokenizer = AutoTokenizer.from_pretrained(tok_path)
        self.detection_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.detection_model.to(device)

    def score_binary(self, y_1, y_2):
        inputs = self.detection_tokenizer(y_1, y_2, return_tensors="pt", padding=True).to(device)
        outputs = self.detection_model(**inputs)
        scores = outputs.logits.softmax(dim=-1)
        # Return probabilites and scores for not paraphrase and paraphrase
        return scores.T[0].item(), scores.T[1].item()

def run_eval(args, model, datasets['test']):
    # reference examples ROGUE
    # consistency
    predictions = ["hello there", "general kenobi"]
    references = ["hello there", "general kenobi"]
    rogue_results = rouge.compute(predictions=predictions,
                            references=references)
    