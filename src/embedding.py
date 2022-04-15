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

class Embedder(nn.Module):
  def __init__(self, trainset):
    self.corpus = "glove.6B.100d.txt"
    glove_frame = pd.read_csv('/Volumes/Untitled/glove/glove.6B.100d.txt',
                              sep=" ", quoting=3, header=None, index_col=0)
    self.embeddings = {trainset.index(w): torch.tensor(vec.values)
                         for w, vec in glove_frame.T.items()}
    self.trainset = trainset

    self.layer = nn.Embedding(len(trainset.vocab), 100)

  def word_embed(self, word_idx):
    try: vect = self.embeddings[word_idx]
    except KeyError:
      vect = torch.rand(100)
      self.embeddings[word_idx] = vect

    vect += self.layer(word_idx)
    return vect

  def pos_embed(self, pos_idx):
    one_hot = torch.functional.one_hot(pos_idx)
    return one_hot
