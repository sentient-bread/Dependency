from torch import nn
from embedding import MLP
import torch
from icecream import ic
from data import *
from settings import *
from pos import *

class EdgeScorer(nn.Module):
  def __init__(self, output_size, vocab_size,
               word_embedding_dimension, hidden_size,
               pos_embedding_dimension, num_pos_tags, pos_tagger,
               vocab=None, words_to_indices=None):
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
    self.bias_arc = nn.Linear(output_size, 1, bias=False)
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
    # Wa_Hd = (H_dep (W_arc)^T)^T
    #       =  W_arc (H_dep)^T

    Hh_Wa_Hd = torch.matmul(H_head, Wa_Hd)
    # Hh_Wa_Hd : [batch_size, sentence_length, sentence_length]
    # Hh_Wa_Hd = H_head W_arc (H_dep)^T

    Hh_B = self.bias_arc(H_head)
    # Hh_B : [batch_size, sentence_length, 1]
    Hh_B = Hh_B.repeat(1, 1, Hh_Wa_Hd.size(dim=2))
    # Hh_B : [batch_size, sentence_length, sentence_length]
    # Hh_B = H_head (B_arc)^T [for all dependents]

    scores = Hh_Wa_Hd + Hh_B
    # scores[i][j] = prob of w[i] being head of w[j]
    # scores = H_head W_arc (H_dep)^T + H_head (B_arc)^T

    return scores

  def predict(self, batch):
    scores = self.forward(batch)

    # Most likely head for each dependent
    return scores.argmax(dim=1)

def train_epoch(model, optimizer, loss_fun, dataloader):
  avg_loss = 0
  for batch in dataloader:
    pred = model(batch[:, 0, :])
    targ = batch[:, 2, :]
    loss = loss_fun(pred, targ)

    loss.backward()
    ic(loss)
    avg_loss += loss
    optimizer.step()
    optimizer.zero_grad()

  avg_loss /= len(dataloader)
  ic(avg_loss)

def train(model, optimizer, loss_fun, dataset, num_epochs):
    for epoch in range(num_epochs):
      print(f"{epoch+1}")
      dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)
      train_epoch(model, optimizer, loss_fun, dataloader)
      print("-------")
      torch.save(model, EDGESCORER_MODEL_PATH)

def train_edgescorer(train_path, num_epochs):
  train_dataset = Dataset(from_vocab=False, file_path=train_path, vocab=None, words_to_indices=None)

  postagger = torch.load(POS_MODEL_PATH, map_location=torch.device('cpu'))
  postagger.eval()
  for param in postagger.parameters(): # Freeze the POS tagger
      param.requires_grad = False

  model = EdgeScorer(100, len(train_dataset.vocab),
                     100, 50,
                     100, 18, postagger,
                     vocab=train_dataset.vocab,
                     words_to_indices=train_dataset.words_to_indices)

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  loss_fun = torch.nn.CrossEntropyLoss(ignore_index=PAD_DEPENDENCY)

  train(model, optimizer, loss_fun, train_dataset, num_epochs)

def test_edgescorer(model, test_path):
  test_dataset = Dataset(from_vocab=True, file_path=test_path, vocab=model.vocab, words_to_indices=model.words_to_indices)
  dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

  total, matches = 0, 0
  for batch in dataloader:
      inp = batch[:,0,:]
      pred = model.predict(inp)
      targ = batch[:,2,:]
      comparison = torch.flatten(torch.stack((model.predict(inp), targ), dim=2),
                                 start_dim=0, end_dim=1)
      
      for pred_head, real_head in comparison:
          if (real_head == -1): continue
          total += 1
          if (pred_head == real_head): matches += 1
      
  return matches/total
