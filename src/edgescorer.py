from torch import nn
from embedding import MLP
import torch
from icecream import ic

class EdgeScorer(nn.Module):
  def __init__(self, output_size, vocab_size,
               word_embedding_dimension, hidden_size,
               pos_embedding_dimension, num_pos_tags, pos_tagger,
               sentence_length, vocab=None, words_to_indices=None):
    nn.Module.__init__(self)

    self.vocab = vocab
    self.words_to_indices = words_to_indices
    self.pos_tagger = pos_tagger
    # Object of POSTag class

    self.word_embedding_layer = nn.Embedding(vocab_size, word_embedding_dimension)
    self.pos_embedding_layer = nn.Embedding(num_pos_tags, pos_embedding_dimension)

    self.lstm = nn.LSTM(word_embedding_dimension+pos_embedding_dimension, hidden_size,
                        bidirectional=True,batch_first=True)

    self.head_encoder = MLP(hidden_size*2, output_size)
    self.dep_encoder = MLP(hidden_size*2, output_size)
    # *2 because the LSTM is bidirectional

    self.weight_arc = nn.Linear(output_size, output_size, bias=False)
    self.bias_arc = nn.Linear(output_size, sentence_length, bias=False)
    # Treat these as just matrices, not NN layers
    # It's just to make training and mat mult convenient afaiu
    # Probabilities = H_head @ (W_arc @ H_dep + B_arc)

  def forward(self, batch):
    words_embedded = self.word_embedding_layer(batch)

    pos_tags_probabilities = self.pos_tagger(batch)
    pos_tags = self.pos_tagger.predict(pos_tags_probabilities)
    pos_embedded = self.pos_embedding_layer(pos_tags)

    lstm_inputs = torch.cat((words_embedded, pos_embedded), dim=2)
    # Concatenate each word's embedding to its POS tag embedding

    lstm_outputs, (last_hidden_state, last_cell_state) = self.lstm(lstm_inputs)

    H_head = self.head_encoder(lstm_outputs)
    H_dep = self.dep_encoder(lstm_outputs)
    # Head and dependent representations for all words
    # H_head : [batch_size, sentence_length, output_size]
    # H_dep : [batch_size, sentence_length, output_size]

    # Following lines are all matrix multiplications
    Wa_Hd = self.weight_arc(H_dep).transpose(1,2)
    # Wa_Hd : [batch_size, output_size, sentence_length]
    Hh_Wa_Hd = torch.matmul(H_head, Wa_Hd)
    # Hh_Wa_Hd : [batch_size, sentence_length, sentence_length]

    Hh_B = self.bias_arc(H_head)
    # Hh_B : [batch_size, sentence_length, sentence_length]

    scores = Hh_Wa_Hd + Hh_B
    # scores[i][j] = prob of w[i] being head of w[j]

    # We want scores[i][j] = prob of w[j] being head of w[i]
    scores = scores.transpose(1,2)

    return scores

  def predict(self, batch):
    scores = self.forward(batch)

    # Most likely head for each dependent
    return scores.argmax(dim=2)
