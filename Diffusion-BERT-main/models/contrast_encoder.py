import torch
from torch import nn
import numpy as np
import math
from transformers import AutoTokenizer, AutoModel, AutoConfig
import random

tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
config = AutoConfig.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

class Contrast_Encoder(nn.Module):
    def __init__(
        self,
        proj_size=128,
        config_name='princeton-nlp/sup-simcse-bert-base-uncased',
    ):
        super().__init__()
        # Only use 6 layers' bert as basic block
        self.cfg = config
        self.passage_encoder = model
        self.device = self.passage_encoder.device

        # projector
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, proj_size)
        )

    def get_embeds(self, input_ids):
        return self.passage_encoder.word_embedding(input_ids)

    def forward(self, input_ids, attention_mask=None):
        logits = self.passage_encoder(input_ids, attention_mask=attention_mask)
        logits = logits['last_hidden'] + 0 * logits['pooler_output'][:, None, :]
        proj = self.output_proj(logits)
        return logits
    
def word_freq_mask(word_freq, input_ids):
    """
    args:
    word_freq:the frequence of the word appeared in the whole text, the default word_freq is counted on lm1b datasets
    input_ids:the tensor of input from bert tokenizer
    
    for word_freq occored more that 300000 times or less that 1500, we directly mask it.
    for word_freq occored between 50000 and 300000 times, we choose 30% to mask
    for word_freq occored between 8000 and 50000 times, we choose 10% to mask
    """
    mask = torch.ones_like(input_ids)
    batch_size = input_ids.size()[0]
    for i in range(batch_size):
        for j in range(len(input_ids[i])):
            if word_freq[input_ids[i][j]] > 300000 or word_freq[input_ids[i][j]] < 1500:
                mask[i][j] = 0
            if word_freq[input_ids[i][j]] > 50000 and word_freq[input_ids[i][j]] <= 300000:
                rand = random.randint(1, 10)
                if rand <= 3:
                    mask[i][j] = 0
            if word_freq[input_ids[i][j]] >= 1500 and word_freq[input_ids[i][j]] <= 50000:
                rand = random.randint(1, 10)
                if rand == 1:
                    mask[i][j] = 0
    return mask
