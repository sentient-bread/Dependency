import torch
from torch import nn
from data import *
from settings import *

CHARACTER_LSTM_LAYERS = 1

class CharacterModel(nn.Module):

    def __init__(self, embedding_size, input_size, hidden_size, final_embedding_dim, vocab_size, max_sequence_length, character_vocab=None, character_to_indices=None):
        super().__init__()
        self.embedding_size = embedding_size
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.character_vocab = character_vocab
        self.character_to_indices = character_to_indices
        self.hidden_size = hidden_size
        self.max_seq_length = max_sequence_length
        self.window_size = int(self.max_seq_length * (0.5))
        self.final_embedding_dim = final_embedding_dim

        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)

        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=CHARACTER_LSTM_LAYERS, batch_first=True)

        self.attention_vector = nn.Linear(self.window_size, 1)
        # trainable attention vector that will do a weighted sum of hidden vectors

        self.linear_layer_W = nn.Linear(self.window_size+self.hidden_size, 
                                        final_embedding_dim)

        self.layer_for_distribution = nn.Linear(final_embedding_dim, self.vocab_size)

        self.softmax_layer = nn.Softmax(dim=0)

    def forward(self, batch):
        #ic(batch)
        # ic(batch.size()) # Gives: [batch_size x window_size]
        embedding_vectors = self.embedding_layer(batch)
        # ic(embedding_vectors.size()) # Gives: [batch_size x window_size x embedding_dimension]
        hidden_vectors, (last_hidden_state, last_cell_state) = self.lstm(embedding_vectors)
        # ic(hidden_vectors.size()) # Gives: [batch_size x window_size x hidden_size]


        hidden_vectors_swapped = torch.swapaxes(hidden_vectors, 1, 2)
        # dimensions: [batch_size x hidden_size x window_size]
        """
        why swap?
        The repeated attention vector is a layer that will be applied as 1 x seq_len
        the vector we had received was batch_size x seq_len x hidden_size
        we kinda transposed it on a plane to get batch_size x hidden_size x seq_len
        in the next line there will be kinda a matrix multiplication that will transpose the layer to seq_len x 1
        and apply the tansformation of repeated_attention_vector x layer across batch
        """
        #ic(hidden_vectors_swapped.size())
        attended_vectors = self.attention_vector(hidden_vectors_swapped)
        #ic(attended_vectors.size()) # gives [batch_size x hidden_size x 1]

        # in other words, we now have the combined vector by attention

        # As per the formula attented_vectors = H w^{attn}

        a = self.softmax_layer(attended_vectors)
        #ic(a.size())

        h_tilde = torch.matmul(hidden_vectors, a)
        # in the formula this looks like 
        # h_tilde = H^{transpose} a
        # hidden_vectors = H^{transpose} basically

        #ic(h_tilde.size()) # batch_size x seq_len x 1
        h_tilde_squeeze = h_tilde.squeeze(dim=2)
        last_cell_state_squeeze = last_cell_state.squeeze(dim=0)
        #ic(h_tilde_squeeze.size(), last_cell_state_squeeze.size())
        #ic(last_cell_state, last_hidden_state)
        v_hat = self.linear_layer_W(
                            torch.concat(
                                (h_tilde_squeeze, last_cell_state_squeeze)
                                , 1))
        #ic(v_hat.size())
        return v_hat

# testing shit 



# for batch in dataloader:
#     model(batch)



def train_epoch(model, optimizer, loss_fun, dataloader):

    for batch in dataloader:
        left = 0
        right = model.window_size
        ic(batch.size())
        counter = 0
        loss_total = 0
        perplexity_total = 1
        while right < batch.size()[1]-1:
            window = batch[:, left:right]
            out = model(window)
            logits = model.layer_for_distribution(out)
            # ic(logits.size())
            probability = model.softmax_layer(logits)
            # ic(probability)
            loss = loss_fun(logits, batch[:, right])
            loss_total += loss.item()
            perplexity_total *= torch.exp(loss).item()
        # assuming that the loss function is cross entropy loss
        # out is a set embeddings making sentences that make batches
        # on the comparison we just give it the pos tags

            loss.backward()
            # ic(loss)
            optimizer.step()
            optimizer.zero_grad()
            left += 1
            right += 1
            counter += 1
        ic(loss_total / counter)
        ic(perplexity_total ** (1/counter))

# train_epoch(model, optimizer, loss_fun, dataloader)

def train(model, optimizer, loss_fun, dataset, num_epochs):

    for epoch in range(num_epochs):
        print(f"{epoch+1}")
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE)
        train_epoch(model, optimizer, loss_fun, dataloader)
        print("-------")
        torch.save(model, CHARACTER_MODEL_PATH)

def train_character_model(train_path, num_epochs):

    train_dataset = DatasetCharacter(False, file_path=train_path)


    dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=69)

    model = CharacterModel(100, 
                            len(train_dataset.character_vocab), 
                            400, 
                            100, 
                            len(train_dataset.character_vocab), 
                            train_dataset.length_longest_character_sequence,
                            character_vocab=train_dataset.character_vocab,
                            character_to_indices=train_dataset.character_to_indices).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss_fun = torch.nn.CrossEntropyLoss()

    train(model, optimizer, loss_fun, train_dataset, num_epochs)
  # train_epoch(model, None, None, test_dataloader)

# train_character_model('../data/UD_English-Atis/en_atis-ud-train.conllu', 20)

def test_character_model(model, test_path):

    test_dataset = DatasetCharacter(True, file_path=test_path,
                                    character_vocab=model.character_vocab
                                    , character_to_indices=model.character_to_indices)


    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=100)
    loss_fun = nn.CrossEntropyLoss()
    for batch in dataloader:
        left = 0
        right = model.window_size
        ic(batch.size())
        counter = 0
        loss_total = 0
        perplexity_total = 1
        ic(right)
        while right < batch.size()[1]-1:
            window = batch[:, left:right]
            out = model(window)
            logits = model.layer_for_distribution(out)
            # ic(logits.size())
            probability = model.softmax_layer(logits)
            # ic(probability)
            loss = loss_fun(logits, batch[:, right])
            loss_total += loss.item()

        # assuming that the loss function is cross entropy loss
        # out is a set embeddings making sentences that make batches
        # on the comparison we just give it the pos tags

            left += 1
            right += 1
            counter += 1

        avg_loss = loss_total / counter
        ic(avg_loss)
        ic(torch.exp(torch.tensor(avg_loss))) #perplexity

model_load = torch.load(CHARACTER_MODEL_PATH).to(DEVICE)
test_character_model(model_load, '../data/UD_English-Atis/en_atis-ud-train.conllu')
