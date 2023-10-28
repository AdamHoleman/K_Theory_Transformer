import torch
import torch.nn as nn
import torch.nn.functional as F


class linear_probe(nn.Module):
  def __init__(self, emb_dim, seq_len, label_size, label_dims=1):
    """
    label_size is the target size of our probe, which equals num_classes*label_dims
    """
    super().__init__()
    if label_dims > 1:
      self.label_shape = (label_size//label_dims, label_dims)
    else:
      self.label_shape = (label_size, -1)

    self.act_size = emb_dim*seq_len
    self.proj = nn.Linear(self.act_size, label_size)
    #initialize weights
    self.proj.weight.data.normal_(mean=0.0, std=0.02)
    self.proj.bias.data.zero_()

  def forward(self, acts, label=None):
    batch_size = acts.size(0)

    logits = self.proj(acts.view(batch_size, self.act_size))
    logits = logits.view(batch_size, self.label_shape[0], self.label_shape[1])

    if label is not None:
      return logits, F.cross_entropy(logits, label)
    else:
      return logits