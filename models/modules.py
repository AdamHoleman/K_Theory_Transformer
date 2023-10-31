import torch
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F


def encode_single_pos(pos, emb_dim):
    inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, emb_dim, 2).float() / emb_dim)
    )

    even_idx = torch.arange(0, emb_dim, 2)
    odd_idx = torch.arange(1, emb_dim, 2)

    target = torch.zeros(emb_dim) #initialize the tensor

    even_enc = torch.sin(pos*torch.ones(1, emb_dim // 2) * inv_freq)
    odd_enc = torch.cos(pos*torch.ones(1, emb_dim // 2) * inv_freq)

    target[even_idx] = even_enc
    target[odd_idx] = odd_enc
    return target

def positional_encoding(num_pos, emb_dim):
    """
    creates a tensor of size (num_pos, emb_dim) with sinusoidal positions
    """

    encoding = torch.zeros(num_pos, emb_dim)
    for pos in range(num_pos):
        encoding[pos] = encode_single_pos(pos, emb_dim)

    return encoding.unsqueeze(0)

def construct_mask(seq):
    #unbatched input
    batch_size, seq_len = seq.size()
    mask = (1- torch.triu(torch.ones((1, seq_len, seq_len)), diagonal = 0)).bool()
    return torch.transpose(mask, 1, 2)


class attention_block(nn.Module):
  """
  This module composes a self-attention layer and a decoder attention layer as in 'attention is all you need'.
  Includes the residual connections and layernorms.
  """

  def __init__(self, emb_dim, n_heads, vocab_size, seq_len):
    super().__init__()
    self.sa = nn.MultiheadAttention(emb_dim, n_heads,
                                    batch_first = True,
                                    dropout = 0.1) 
    self.norm1 = nn.LayerNorm(emb_dim)
    
    self.att = nn.MultiheadAttention(emb_dim, n_heads,
                                     batch_first = True,
                                     dropout = 0.1) 
    self.norm2 = nn.LayerNorm(emb_dim)

  def forward(self, src, trg, trg_mask):
    att, _ = self.sa(trg,trg,trg, attn_mask = trg_mask)
    trg = self.norm1(trg + att)

    att, _ = self.att(trg,src,src)
    trg = self.norm2(trg + att)

    return trg


class MLP(nn.Module):
  def __init__(self, emb_dim, dim_feedforward):
    super().__init__()
    self.emb = nn.Linear(emb_dim, dim_feedforward)
    self.relu = nn.ReLU()
    self.proj = nn.Linear(dim_feedforward, emb_dim)
    
  def forward(self, x, return_act = False):
    act = self.emb(x)
    act = self.relu(act)

    if return_act:
      return act

    return self.proj(act)




class probeable_decoder_model(nn.Module): 
  """
  This is a standard decoding transformer, but the forward pass has been altered to return the activations at the desired layer. 
  We allow for probing at the residual stream after the attention mechanism has been applied and at the hidden layer of the MLP. 
  Since each layer has 2 probe points, when calling the forward pass probe_depth should be between 0 and 2*num_layers+1.
  Setting probe_depth=0 (default) just runs the transformer as normal.
  """

  def __init__(self, emb_dim, n_heads, vocab_size, seq_len, num_layers, dim_feedforward = 2048):
    super().__init__()
    self.n_heads = n_heads
    self.emb_dim = emb_dim
    self.seq_len = seq_len

    self.trg_emb = nn.Embedding(vocab_size, emb_dim)
    self.src_emb = nn.Embedding(vocab_size, emb_dim)

    self.attention_list = nn.ModuleList()
    self.mlp_list = nn.ModuleList()
    self.normalizations = nn.ModuleList()
    self.num_layers = num_layers

    for layer in range(num_layers):
      """
      Create attention, MLP, and layernorm modules for each layer
      """
      self.attention_list.append(attention_block(emb_dim, n_heads,
                                                 vocab_size, seq_len))  
      self.mlp_list.append(MLP(emb_dim, dim_feedforward)) 
      self.normalizations.append(nn.LayerNorm(emb_dim))

    self.trg_out = nn.Linear(emb_dim, vocab_size)

  def forward(self, src, trg, probe_depth=0):

    #Embed and add positional encodings (moving them to the same device as the input)
    trg_pos_enc = positional_encoding(self.seq_len, self.emb_dim).repeat(trg.size()[0],1,1).to(trg.device)
    src_pos_enc = positional_encoding(3, self.emb_dim).repeat(src.size()[0],1,1).to(trg.device)
    trg_mask = construct_mask(trg).repeat(trg.size()[0]*self.n_heads, 1, 1).to(trg.device)

    trg = self.trg_emb(trg)
    out = trg + trg_pos_enc

    src = self.src_emb(src)
    src += src_pos_enc

    #loop through the attention blocks
    for i in range(self.num_layers):

      out = self.attention_list[i](src, out, trg_mask)

      #decrement the depth tracker and check if we've reached the probe layer, in which case we return the activations of the residual stream. 
      probe_depth -= 1
      if probe_depth == 0:
        return out

      #check if we've reached the probe layer, in which case we return the activations of the MLP hidden layer. If not, decrement the depth tracker and move on.
      probe_depth -= 1
      if probe_depth == 0:
        return self.mlp_list[i](out, return_act = True)

      residual = self.mlp_list[i](out)
      out = self.normalizations[i](out + residual)

      #last probe check
      probe_depth -=1
      if probe_depth == 0:
        return out

    out = self.trg_out(out) #we want to output unnormalized logits

    return out



  def predict(self, src):
    #Given an input src = (r,e,p), auto-regressively predict the p^{th}-power torsion summands of K_{2r-1}(F_{p}[x]/x^e)

    batch, _ = src.size()
    trg = torch.zeros(batch, self.seq_len)

    for i in range(self.seq_len):
      logits = self.forward(src, trg) #size (batch, seq_len, vocab_size)
      preds = torch.argmax(out, dim=2)
      trg[:, i] = preds[:, i] #pluck the prediction for the i^{th} token and update the trg sequence

    return trg


    










