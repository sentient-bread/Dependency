import torch
from torch import nn
import pandas as pd

class MLP(nn.Module):
  def __init__(self, input_size, output_size):
    nn.Module.__init__(self)
    self.layer = nn.Linear(input_size, output_size)
    self.activation = nn.ReLU()

  def forward(self, sentence_emb):
    sentence_repr = self.layer(sentence_emb)
    output = self.activation(sentence_repr)
    return output
