from torch import nn
from embedding import MLP

class POSTag(nn.Module):
  def __init__(self, input_size, output_size)
    self.encoder = MLP(input_size, output_size)
    self.linear = nn.Linear(output_size, 17)

  def forward(self, sentence_repr):
    encoded = self.encoder(sentence_repr)
    prob_dists = self.linear(encoded)

    return prob_dists

  def predict(self, sentence_repr):
    prob_dists = self.forward(sentence_repr)

    return prob_dists.argmax(dim=1)