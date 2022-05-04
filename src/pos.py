from torch import nn
from embedding import MLP
from data import *
import torch
from settings import *
from sklearn import metrics


class POSTag(nn.Module):
  def __init__(self, output_size, vocab_size, embedding_dimension, hidden_size, num_pos_tags, vocab=None, words_to_indices=None):

    """
    as per the paper
    output_size = num_pos_tags = 17 (in our case 18)

    we have them as different variables for now, for experimentation

    vocab argument is to be given only during training
    It represents the vocabulary used for training
    similarly for words to indices
    """
    super().__init__()

    self.num_pos_tags = num_pos_tags
    self.vocab = vocab
    self.words_to_indices = words_to_indices
    self.output_size = output_size

    self.embedding_layer = nn.Embedding(vocab_size, embedding_dimension)
    # may be pretrained

    self.lstm = nn.LSTM(embedding_dimension, hidden_size, bidirectional=True, batch_first=True)

    self.classifier = MLP(hidden_size*2, output_size)
    # the multiplied by 2 is because of bilstm
    # the final output will be of hidden_size * 2

    self.w_layer = nn.Linear(output_size, num_pos_tags, bias=True)

  def forward(self, batch):

    """
    assuming a batch_size * sentence_length tensor
    each row represents a sentence
    THATS IT
    """
    embedded = self.embedding_layer(batch)

    lstm_outputs, (last_hidden_state, last_cell_state) = self.lstm(embedded)


    classifier_outputs = self.classifier(lstm_outputs)
    # MLP Classifier
    affine_layer_outputs = self.w_layer(classifier_outputs)


    return affine_layer_outputs

  def predict(self, batch):

    #affine_layer_output = self.forward(batch)
    return batch.argmax(dim=2)

  def get_W_transpose_matrix(self):

    w_transpose = self.w_layer.state_dict()['weight']
    return w_transpose



def train_epoch(model, optimizer, loss_fun, dataloader):

  for batch in dataloader:
    out = model(batch[:, 0, :])
    most_likely = model.predict(out)
    out_swapped = torch.swapaxes(out, 1, 2)
    loss = loss_fun(out_swapped, batch[:, 1, :])
    # assuming that the loss function is cross entropy loss
    # out is a set embeddings making sentences that make batches
    # on the comparison we just give it the pos tags

    loss.backward()
    ic(loss)
    optimizer.step()
    optimizer.zero_grad()

def train(model, optimizer, loss_fun, dataset, num_epochs, pos_model_path):

  for epoch in range(num_epochs):
    print(f"{epoch+1}")
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)
    train_epoch(model, optimizer, loss_fun, dataloader)
    print("-------")
    torch.save(model, pos_model_path)


def train_POS(train_path, num_epochs, pos_model_path):

  train_dataset = Dataset(False, file_path=train_path)
  # test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE)



  model = POSTag(
                100,
                len(train_dataset.vocab),
                100,
                200,
                18,
                vocab=train_dataset.vocab,
                words_to_indices=train_dataset.words_to_indices).to(DEVICE)

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  loss_fun = torch.nn.CrossEntropyLoss()

  train(model, optimizer, loss_fun, train_dataset, num_epochs, pos_model_path)
  # train_epoch(model, None, None, test_dataloader)



def test_model(model, test_file_path):

  test_dataset = Dataset(True, vocab=model.vocab, words_to_indices=model.words_to_indices, file_path=test_file_path)
  # test_dataset initialized with vocabulary taken from loaded model

  dataloader = torch.utils.data.DataLoader(test_dataset)

  total_batches = 0
  sum_accuracy = 0

  for batch in dataloader:
    pred = model(batch[:, 0, :])
    pred = model.predict(pred)
    total_tokens = 0
    accurate_tokens = 0

    for i in range(pred.shape[-1]):
      if batch[:, 1, i] != 17:
        if pred[:, i] == batch[:, 1, i]:
          accurate_tokens += 1
        total_tokens += 1
    accuracy = accurate_tokens / total_tokens
    # embeddings = model.get_pos_embedding(batch[:, 1, :])
    # model.output_size = 100
    # matrix = model.get_W_transpose_matrix()
    sum_accuracy += accuracy
    total_batches += 1

  average_accuracy = sum_accuracy / total_batches

  ic(average_accuracy)


def all_metrics(model, test_file_path):
  test_dataset = Dataset(True, vocab=model.vocab, words_to_indices=model.words_to_indices, file_path=test_file_path)
  # test_dataset initialized with vocabulary taken from loaded model

  dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

  tag_dict = {i: tag for tag, i in test_dataset.tags_to_indices.items()}

  pred_all = []
  gold_all = []
  for i, batch in enumerate(dataloader):
    pred = model(batch[:, 0, :])
    pred = model.predict(pred)
    pred = torch.flatten(pred)
    pred = [tag_dict[i.item()] for i in pred]
    
    gold = batch[:, 1, :]
    gold = torch.flatten(gold)
    gold = [tag_dict[i.item()] for i in gold]

    print(f"Batch {i+1}")
    print(metrics.classification_report(gold, pred))
    
    pred_all += pred
    gold_all += gold

  print("Overall")
  print(metrics.classification_report(gold_all, pred_all))
    


# train_path = '../data/UD_English-Atis/en_atis-ud-train.conllu'
# train_path = '../data/hindi/hi_hdtb-ud-train.conllu'
# train_path = '../data/sanskrit/sa_vedic-ud-train.conllu'

# train_POS(train_path, 20, POS_MODEL_PATH)
# train_POS(train_path, 20, POS_MODEL_PATH_HINDI)
# train_POS(train_path, 20, POS_MODEL_PATH_SANSKRIT)

# load_model = torch.load(POS_MODEL_PATH).to(DEVICE)
# load_model = torch.load(POS_MODEL_PATH_HINDI).to(DEVICE)
# load_model = torch.load(POS_MODEL_PATH_SANSKRIT).to(DEVICE)

# ic(load_model.vocab)

# test_file_path = "../data/UD_English-Atis/en_atis-ud-test.conllu"
# test_file_path = '../data/hindi/hi_hdtb-ud-test.conllu'
# test_file_path = '../data/sanskrit/sa_vedic-ud-test.conllu'

# test_model(load_model, test_file_path)

# all_metrics(load_model, test_file_path)
