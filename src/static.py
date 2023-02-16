import torch

DATA_DIR = '../data'
MODEL_DIR = '../models'

PARAPHRASE_MODEL = ''

MODELS = {
    'gpt': {
        'small': 'distilgpt2'
    },
    't5': {
        'small': 't5-small'
    }
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
