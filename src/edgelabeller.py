import torch
from torch import nn
from embedding import MLP
from icecream import ic
from settings import *
from data import *
from pos import POSTag

class EdgeLabeller(nn.Module):
  def __init__(self,
      output_size,
      vocab_size,
      word_embedding_dimension,
      hidden_size,
      pos_embedding_dimension,
      num_pos_tags,
      pos_tagger,
      vocab=None,
      words_to_indices=None,
      relations_to_indices=RELATIONS_TO_INDICES,
      labels=UNIVERSAL_DEPENDENCY_LABELS):
    super().__init__()

    self.word_embedding_layer = nn.Embedding(vocab_size, word_embedding_dimension)
    self.pos_embedding_layer = nn.Embedding(num_pos_tags, pos_embedding_dimension)

    self.lstm = nn.LSTM(word_embedding_dimension + pos_embedding_dimension, hidden_size,
                        bidirectional=True,batch_first=True)

    self.head_encoder = MLP(hidden_size * 2, output_size)
    self.dep_encoder = MLP(hidden_size * 2, output_size)

    self.both_weight_rel = nn.Bilinear(output_size, output_size, len(labels), bias=False)
    # the U^(rel) matrix in the paper
    self.either_weight_rel = nn.Linear(output_size * 2, len(labels), bias=True)
    # The W^(rel) matrix in the original paper
    # The bias layer is encapsulated in the second matrix

  def forward(self, batch):
    # Assuming that the batch has three lists: words, pos, heads
    WORDS, POS, HEADS = 0, 1, 2
    ic(batch.shape)

    words_embedded = self.word_embedding_layer(batch[:, WORDS, :])
    heads_indices = torch.tensor(batch[:, HEADS, :])
    # Heads indices must be of the form
    # [ [indices for first sentence in batch]
    #   [indices for second sentence in batch]
    #                ...
    #   [indices for n'th sentence in batch] ]
    # NOTE: Make sure heads_indices handles BOS and padding

    pos_tags_probabilities = self.pos_tagger(batch)
    pos_tags = self.pos_tagger.predict(pos_tags_probabilities)
    pos_embedded = self.pos_embedding_layer(pos_tags)

    lstm_inputs = torch.cat((words_embedded, pos_embedded), dim=2)
    # Concatenate each word's embedding to its POS tag embedding

    lstm_outputs, (last_hidden_state, last_cell_state) = self.lstm(lstm_inputs)

    H_dep = self.dep_encoder(lstm_outputs)

    # Now we must first reorder the sentences in the batch of heads so
    # interaction weight matrix U^(rel) actually trains the weights between the
    # heads and the dependents instead of the words and themselves.
    H_head_raw = self.head_encoder(lstm_outputs)

    # The dimension in which we index will be along each sentence in the batch
    H_head = torch.gather(H_head_raw, heads_indices, dim=1)

    # The first term intuitively relates to the probability of a specific label
    # given the interaction between *both* the head and the label. This is the
    # U^(rel) matrix in the paper.
    Hh_Ur_Hd = self.both_weight_rel(H_head, H_dep)
    ic(Hh_Ur_Hd.shape)

    # The second term intuitively relates to the probability of a specific
    # label given *either* the head or the label's information. This it the
    # W^(rel) matrix in the paper.

    # The dimension we concatenate along are the embeddings themselves.
    Wr_Hr_Hd = self.either_weight_rel(torch.cat(H_head, H_dep, dim=2))
    ic(Wr_Hr_Hd.shape)

    scores = Hh_Ur_Hd + Wr_Hr_Hd
    ic(scores.shape)
    return scores

  def predict(self, batch):
    scores = self.forward(batch)
    return scores.argmax(dim=1)

def train_epoch(model, optimizer, loss_fun, dataloader):
  avg_loss = 0
  for batch in dataloader:
    predictions = model(batch)
    target = batch[:, 3, :]
    loss = loss_fun(predictions, target)

    loss.backward()
    ic(loss)
    avg_loss += loss
    optimizer.step()
    optimizer.zero_grad()

  avg_loss /= len(dataloader)
  ic(avg_loss)

def train(model, optimizer, loss_fin, dataset, num_epochs):
  for epoch in range(num_epochs):
    print(f"{epoch+1}")
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)
    train_epoch(model, optimizer, loss_fun, dataloader)
    print("-------")
    torch.save(model, EDGELABELLER_MODEL_PATH)

def train_edgelabeller(train_path, num_epochs):
  train_dataset = Dataset(from_vocab=False, file_path=train_path, vocab=None, words_to_indices=None)

  postagger = torch.load(POS_MODEL_PATH, map_location=torch.device('cpu'))
  postagger.eval()
  for param in postagger.parameters():
    param.requires_grad = False

  model = EdgeLabeller(100, len(train_dataset.vocab),
                       100, 50,
                       100, 18, postagger,
                       vocab=train_dataset.vocab,
                       words_to_indices=train_dataset.words_to_indices)

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  loss_fun = nn.CrossEntropyLoss(ignore_index=PAD_DEPENDENCY)

  train(model, optimizer, loss_fun, train_dataset, num_epochs)
