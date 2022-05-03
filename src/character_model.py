import torch
from torch import nn
from data import *
from settings import *

CHARACTER_LSTM_LAYERS = 1

class CharacterModel(nn.Module):

    def __init__(self, embedding_size, input_size, hidden_size, character_level_embedding, vocab_size, max_sequence_length, character_vocab=None, character_to_indices=None):
        super().__init__()
        self.embedding_size = embedding_size
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.character_vocab = character_vocab
        self.character_to_indices = character_to_indices
        self.hidden_size = hidden_size
        self.max_seq_length = max_sequence_length
        self.window_size = int(self.max_seq_length * (0.5))
        self.character_level_embedding = character_level_embedding

        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)

        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=CHARACTER_LSTM_LAYERS, batch_first=True)

        self.attention_vector = nn.Linear(self.hidden_size, 1)
        # trainable attention vector that will do a weighted sum of hidden vectors

        self.linear_layer_W = nn.Linear(self.hidden_size+self.hidden_size, 
                                        character_level_embedding)

        self.layer_for_distribution = nn.Linear(character_level_embedding, self.vocab_size)

        self.softmax_layer = nn.Softmax(dim=0)

    def forward(self, batch):
        #ic(batch)
        ic(batch.shape)
        # ic(batch.size()) # Gives: [batch_size x num_word x num_characters]
        embedding_vectors = self.embedding_layer(batch)
        # ic(embedding_vectors.shape) # batch_size x num_words x num_characters x embedding_size


        ic(embedding_vectors.view(embedding_vectors.shape[0]*embedding_vectors.shape[1], embedding_vectors.shape[2], embedding_vectors.shape[3]).shape)
        lstm_inputs = embedding_vectors.view(embedding_vectors.shape[0]*embedding_vectors.shape[1], 
                                            embedding_vectors.shape[2], 
                                            embedding_vectors.shape[3])

        # Above line is important because of how lstm takes inputs
        # it can only take 3 dimensional tensors
        # we want it to treat a word as a sequence and that is the only thing we care for
        # we originally had [batch_size x num_words x num_characters x embedding_size]
        # we merge first two dimensions: [batch_size * num_words x num_characters x embedding_size]

        hidden_vectors, (last_hidden_state, last_cell_state) = self.lstm(lstm_inputs)
        ic(hidden_vectors.size()) # Gives: [batch_size * num_words x num_characters x hidden_size]


        attended_vectors = self.attention_vector(hidden_vectors)
        ic(attended_vectors.size()) # gives [batch_size * num_words x word_length x 1]

        a = self.softmax_layer(attended_vectors)
        ic(a.size()) # [batch_size * num_words x word_length x 1]

        hidden_vectors_swapped = torch.swapaxes(hidden_vectors, 1, 2) # H^T
        # dimensions: [batch_size * num_words x hidden_size x num_characters]

        h_tilde = torch.matmul(hidden_vectors_swapped, a)
        # in the formula this looks like
        # h_tilde = H^{transpose} a

        ic(h_tilde.size()) # num_words * num_words x hidden_size x 1
        h_tilde_squeeze = h_tilde.squeeze(dim=2)
        last_cell_state_squeeze = last_cell_state.squeeze(dim=0)
        # ic(h_tilde_squeeze.size(), last_cell_state_squeeze.size())

        v_hat = self.linear_layer_W(
                            torch.concat(
                                (h_tilde_squeeze, last_cell_state_squeeze), 1
                                )
                            )
        ic(v_hat.size()) # [batch_size * num_words x embedding_size]

        # unmerging dimensions

        v_hat_unmerged = v_hat.view(batch.shape[0], batch.shape[1], self.embedding_size)
        # ic(v_hat_unmerged.shape) #[batch_size x num_words x embedding_size]
        # the forward method returns something like te output of an embedding layer
        # therefore this can be used directly

        return v_hat_unmerged






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

    for batch in dataloader:
        model(batch)

    #train(model, optimizer, loss_fun, train_dataset, num_epochs)
  # train_epoch(model, None, None, test_dataloader)

train_character_model('../data/UD_English-Atis/en_atis-ud-train.conllu', 20)

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

# model_load = torch.load(CHARACTER_MODEL_PATH).to(DEVICE)
# test_character_model(model_load, '../data/UD_English-Atis/en_atis-ud-train.conllu')
