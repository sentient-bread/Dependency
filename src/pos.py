from torch import nn
from embedding import MLP
from data import *
import torch
from settings import *



class POSTag(nn.Module):
  def __init__(self, output_size, vocab_size, embedding_dimension, hidden_size, num_pos_tags):

    """
    as per the paper
    output_size = num_pos_tags = 17 (in our case 18)

    we have them as different variables for now, for experimentation
    """
    super().__init__()
    self.embedding_layer = nn.Embedding(vocab_size, embedding_dimension)
    # may be pretrained

    self.lstm = nn.LSTM(embedding_dimension, hidden_size, bidirectional=True, batch_first=True)

    self.classifier = MLP(hidden_size*2, output_size)
    # the multiplied by 2 is because of bilstm
    # the final output will be of hidden_size * 2

    self.linear = nn.Linear(output_size, num_pos_tags, bias=True)

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

    affine_layer_outputs = self.linear(classifier_outputs)


    return affine_layer_outputs

  def predict(self, batch):

    #affine_layer_output = self.forward(batch)
    return batch.argmax(dim=2)

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

def train(model, optimizer, loss_fun, dataset, num_epochs):

  for epoch in range(num_epochs):
    print(f"{epoch+1}")
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)
    train_epoch(model, optimizer, loss_fun, dataloader)
    print("-------")
    torch.save(model.state_dict(), POS_MODEL_PATH)



test_dataset = Dataset(trainfile='../data/UD_English-Atis/en_atis-ud-test.conllu')
# test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE)



model = POSTag(
              18,
              len(test_dataset.vocab),
              100,
              50,
              18).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

loss_fun = torch.nn.CrossEntropyLoss()

train(model, optimizer, loss_fun, test_dataset, 100)
# train_epoch(model, None, None, test_dataloader)
