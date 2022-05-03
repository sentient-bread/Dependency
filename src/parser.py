import torch
from torch import nn

from data import *
from embedding import *
from character_model import *
from edgescorer import *
from edgelabeller import *
from pos import *

class Parser(nn.Module):
  def __init__(self, char_emb_size, char_hid_size, char_vocab,
                     word_emb_size, vocab, pos_emb_size,
                     upos_tagger, xpos_tagger,
                     pars_hid_size, mlp_size):
    self.character_embedding = CharacterModel(char_emb_size, char_hid_size, word_emb_size, char_vocab,
                                              {char: idx for idx, char in enumerate(char_vocab)})
    self.trained_embedding = nn.Embedding()
    self.glove_embedding = nn.Embedding()

    self.upos_tagger = upos_tagger
    self.xpos_tagger = xpos_tagger

    self.bilstm = nn.LSTM(word_emb_size + pos_emb_size, pars_hid_size, bidirectional=True, batch_first=True)

    self.edgescorer = EdgeScorer()
    self.edgelabeller = EdgeLabeller()

  def train(self, train_file):
    pass