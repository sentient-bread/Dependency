import torch
from torch import nn
from data import *
from settings import *

CHARACTER_LSTM_LAYERS = 1

class CharacterModel(nn.Module):

    def __init__(self, embedding_size, input_size, hidden_size, final_embedding_dim, vocab_size, max_sequence_length, character_vocab=None, characters_to_indices=None):
        super().__init__()
        self.embedding_size = embedding_size
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.character_vocab = character_vocab
        self.characters_to_indices = characters_to_indices
        self.hidden_size = hidden_size
        self.max_seq_length = max_sequence_length
        self.final_embedding_dim = final_embedding_dim
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)

        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=CHARACTER_LSTM_LAYERS, batch_first=True)

        self.attention_vector = nn.Linear(max_sequence_length, 1)
        # trainable attention vector that will do a weighted sum of hidden vectors

        self.linear_layer_W = nn.Linear(self.max_seq_length+self.hidden_size, 
                                        final_embedding_dim)

        self.softmax_layer = nn.Softmax(dim=1)

    def forward(self, batch):
        ic(batch)
        # ic(batch.size()) # Gives: [batch_size x max_character_sequence_length]
        embedding_vectors = self.embedding_layer(batch)
        # ic(embedding_vectors.size()) # Gives: [batch_size x max_character_sequence_length x embedding_dimension]
        hidden_vectors, (last_hidden_state, last_cell_state) = self.lstm(embedding_vectors)
        # ic(hidden_vectors.size()) # Gives: [batch_size x max_character_seq_length x hidden_size]


        hidden_vectors_swapped = torch.swapaxes(hidden_vectors, 1, 2)
        # dimensions: [batch_size x hidden_size x max_character_seq_length]
        """
        why swap?
        The repeated attention vector is a layer that will be applied as 1 x seq_len
        the vector we had received was batch_size x seq_len x hidden_size
        we kinda transposed it on a plane to get batch_size x hidden_size x seq_len
        in the next line there will be kinda a matrix multiplication that will transpose the layer to seq_len x 1
        and apply the tansformation of repeated_attention_vector x layer across batch
        """
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
        ic(last_cell_state, last_hidden_state)
        v_hat = self.linear_layer_W(
                            torch.concat(
                                (h_tilde_squeeze, last_cell_state_squeeze)
                                , 1))
        ic(v_hat)

# testing shit 


# dataset = DatasetCharacter(False, file_path='../data/UD_English-Atis/en_atis-ud-train.conllu')

# ic(dataset.character_vocab)

# dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=69)

# model = CharacterModel(100, len(dataset.character_vocab), 100, 50, len(dataset.character_vocab), dataset.length_longest_character_sequence).to(DEVICE)

# for batch in dataloader:
#     model(batch)