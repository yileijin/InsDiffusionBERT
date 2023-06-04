from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
from torch import nn
import numpy as np
import math
from models.CrossAttentionTransformers import BasicTransformerBlock

tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
config = AutoConfig.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class Conditional_Encoder(nn.Module):
    def __init__(self, config=config, fix_encoder=True, att_block_num=2):
        super(Conditional_Encoder, self).__init__()
        # time embedding layer
        time_embed_dim = config.hidden_size * 4
        self.time_embed = nn.Sequential(
            nn.Linear(config.hidden_size, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, config.hidden_size),
        )
        
        # position embedding
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.passage_encoder = model
        
        basic_block = BasicTransformerBlock(dim=config.hidden_size,
                                    num_attention_heads=config.num_attention_heads,
                                    attention_head_dim=64,
                                   dropout=config.hidden_dropout_prob)
        self.att_blocks = nn.ModuleList([basic_block for _ in range(att_block_num)])
        
        self.output_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                        nn.Tanh(),
                                        nn.Linear(config.hidden_size, config.vocab_size))
        self.vocab_size = config.vocab_size
        self.device = model.device
        self.fix_encoder = fix_encoder
        self.hidden_size = config.hidden_size
        
    def forward(self, input_ids, timesteps, attention_mask=None):
        emb = self.time_embed(timestep_embedding(timesteps, self.hidden_size))
        seq_length = input_ids.size(1)
        position_ids = self.position_ids[:, : seq_length]
        emb_inputs = self.position_embeddings(position_ids) + emb.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))
        
        if self.fix_encoder == True:
            with torch.no_grad():
                out = self.passage_encoder(input_ids=input_ids, attention_mask=attention_mask)
                hidden = out.last_hidden_state
        else:
            out = self.passage_encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = out.last_hidden_state + 0 * out.pooler_output.unsqueeze(1)
        
        for block in self.att_blocks:
            hidden = hidden + block(hidden)
        conditional_prob = torch.softmax(self.output_proj(hidden), dim=-1)
        return conditional_prob.mean(dim=1, keepdim=True)

