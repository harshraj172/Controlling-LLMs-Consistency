import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftEmbedding(nn.Module):

    def __init__(self, original_emb: nn.Embedding, n_tokens: int = 10):
        """appends learned embedding to original embedding
        Args:
          original_emb (nn.Embedding): original transformer word embedding
          n_tokens (int, optional): number of tokens for task. Defaults to 10.
          init_from_vocab (bool, optional): initalizes from default vocab.
        """
        super().__init__()
        self.name = 'base-embedding'
        self.original_emb = original_emb
        self.n_tokens = n_tokens

        init_prompt_value = self.init_embedding(
            original_emb, n_tokens
        )
        self.soft_prompt = nn.Parameter(
            init_prompt_value, requires_grad=True).to(device)
        print(
            f"Initialized soft prompts with dimension {self.soft_prompt.shape}")

    def init_embedding(self, original_emb, n_tokens, init_from_vocab=True):
        """initializes learned embedding
          either from vocab, random initialization or a custom set of init_text

        Returns:
          torch.float: initialized using original schemes
        """
        if init_from_vocab:
            init_embd = self.original_emb.weight[:n_tokens].clone().detach()
        else:
            rr = 0.5  # random_range
            dimension = original_emb.weight.size(1)
            init_embd = torch.FloatTensor(
                n_tokens, dimension).uniform_(-rr, rr)
            print(f"Initialized embedding with random vectors")
        return init_embd

    def forward(self, tokens):
        raise NotImplementedError

    @classmethod
    def from_saved_embedding(cls, args, original_emb, prompt_path):
        if args.accelerate:
            weights = torch.nn.Parameter(torch.load(prompt_path).half())
        else:
            weights = torch.load(prompt_path)

        num_prompt_tokens = weights.shape[0]
        previous_embed = cls(original_emb, num_prompt_tokens)
        previous_embed.soft_prompt = weights
        print(f"Loaded prompt weights from {prompt_path}")
        return previous_embed

    def save_prompt_embedding(self, save_path, prompt_file):
        prompt_path = os.path.join(save_path, prompt_file)
        torch.save(self.soft_prompt, prompt_path)
        print(f"Saved a soft prompt at {prompt_path}")


class CausalEmbedding(SoftEmbedding):
    """CasualEmbedding is a soft prompt for Causal LMs"""

    def __init__(self, original_emb: nn.Embedding, n_tokens: int = 10):
        super().__init__(original_emb, n_tokens)
        self.name = 'causal-embedding'

    def forward(self, tokens):
        """run forward pass
        Args:
          tokens (torch.long): input tokens before encoding
        Returns:
          torch.float: encoding of text concatenated with learned task specifc embedding

        Reasoning: During the first pass, we are operating in the encoding phase, so we
          modify the input sequence to use the soft prompt.  In subsequent passes, we are
          now operating in the generation phase, so we just process the tokens normally.
          Since generation operates one token at a time, we check whether the sequence
          length is <= 1 token to recognize when we are in the generation phase.
        """
        batch_size, seq_len = tokens.shape
        # use soft prompt unless we are using the autoregressive `.generate()`
        if seq_len > 1:
            input_embed = self.original_emb(tokens[:, self.n_tokens:])
            learned_embed = self.soft_prompt.repeat(batch_size, 1, 1)
            final_embed = torch.cat([learned_embed, input_embed], 1)
        else:
            final_embed = self.original_emb(tokens)
        return final_embed


class Seq2SeqEmbedding(SoftEmbedding):
    """CasualEmbedding is a soft prompt for Seq2Seq LMs"""

    def __init__(self, original_emb: nn.Embedding, n_tokens: int = 10):
        super().__init__(original_emb, n_tokens)
        self.name = 'seq2seq-embedding'

    def forward(self, tokens):
        """run forward pass
        Args:
          tokens (torch.long): input tokens before encoding
        Returns:
          torch.float: encoding of text concatenated with learned task specifc embedding

        Reasoning: During the first pass, we are operating in the encoding phase, which we
          recognize by checking that the first token in the first example contains a negative
          value.  This token_id == -1 since we manually set it as the placeholder earlier.
          When this is not the case, then we are in the generation phase, so we may simply
          proceed as normal with the original embedding.
        """
        if tokens[0][0] < 0:  # if first token is a soft prompt placeholder
            input_embed = self.original_emb(tokens[:, self.n_tokens:])
            learned_embed = self.soft_prompt.repeat(tokens.shape[0], 1, 1)
            final_embed = torch.cat([learned_embed, input_embed], 1)
        else:
            final_embed = self.original_emb(tokens)
        return final_embed


def save_model(save_path, epoch, model):
    ckpt_name = f'model_epoch_{epoch}.pt'
    ckpt_path = os.path.join(save_path, ckpt_name)
    model.save_pretrained(ckpt_path)
    print(f"Saved a model at {ckpt_path}")


def save_soft_prompt(save_path, epoch, embedder):
    ckpt_name = f'softprompt_epoch_{epoch}.pt'
    ckpt_path = os.path.join(save_path, ckpt_name)
    embedder.save_prompt_embedding(save_path, ckpt_name)
    print(f"Saved a soft prompt at {ckpt_path}")
