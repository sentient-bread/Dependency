from torch import nn
from embedding import MLP
import torch
from icecream import ic

class EdgeScorer(nn.Module):
  def __init__(self, input_size, output_size):
    nn.Module.__init__(self)
    self.head_encoder = MLP(input_size, output_size)
    self.dep_encoder = MLP(input_size, output_size)

    self.weight_arc = nn.Linear(output_size, output_size, bias=False)
    self.bias_arc = nn.Linear(output_size, 1, bias=False)

  def forward(self, sentence_repr):
    H_head = self.head_encoder(sentence_repr)
    H_dep = self.dep_encoder(sentence_repr)

    Wa_Hd = self.weight_arc(H_dep).transpose(0,1)
    Hh_Wa_Hd = torch.mm(H_head, Wa_Hd)

    Hh_B = self.bias_arc(H_head)

    scores = Hh_Wa_Hd + Hh_B
    # scores[i][j] = prob of w[i] being head of w[j]

    return scores

  def predict(self, sentence_repr):
    edge_probs = self.forward(sentence_repr)
    heads = edge_probs.transpose(0,1).argmax(dim=1)

    return heads

st = torch.tensor([[1.,2.,3.],[4.,5.,6.]]).float()
es = EdgeScorer(3,5)
