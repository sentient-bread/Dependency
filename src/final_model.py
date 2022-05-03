import torch
from torch import nn
from settings import *
from data import *
from character_model import CharacterModel
from pos import *
from edgescorer import EdgeScorer
from edgelabeller import EdgeLabeller



class Parser(nn.Module):

    def __init__(self,
                head_dep,
                labeller_state,
                vocab_size,
                word_embedding_dimension,
                hidden_size,
                pos_embedding_dimension,
                num_pos_tags,
                pos_tagger,
                character_embedding_dimension,
                pretrained_embedding_weights,
                character_vocab_size,
                character_vocab=None,
                character_to_indices=None,
                vocab=None,
                word_to_indices=None,
                relationships_to_indices=RELATIONS_TO_INDICES,
                labels=UNIVERSAL_DEPENDENCY_LABELS,
                ):
        super().__init__()
        self.pos_tagger: POSTag = pos_tagger
        # get the separately trained pos tagger as a state variable

        self.character_level_model = CharacterModel(character_embedding_dimension,
                                                 hidden_size,
                                                 word_embedding_dimension,
                                                 character_vocab_size,
                                                 character_vocab,
                                                 character_to_indices,
                                                 )

        self.holistic_embedding = nn.Embedding(vocab_size, word_embedding_dimension)

        self.pretrained_embedding_layer = nn.Embedding.from_pretrained(pretrained_embedding_weights)

        self.pos_embeddings = nn.Embedding.from_pretrained(pos_tagger.get_W_transpose_matrix().detach())
        ic(self.pos_embeddings.num_embeddings)
        self.common_lstm = nn.LSTM(word_embedding_dimension + pos_embedding_dimension, hidden_size,
                        bidirectional=True, batch_first=True)

        self.edge_scorer = EdgeScorer(head_dep, vocab_size,
                                    word_embedding_dimension, hidden_size,
                                    pos_embedding_dimension, num_pos_tags,
                                    pos_tagger, self.common_lstm, vocab,
                                    word_to_indices)

        self.edge_labeller = EdgeLabeller(labeller_state, vocab_size,
                                            word_embedding_dimension, hidden_size,
                                            pos_embedding_dimension, 
                                            num_pos_tags,
                                            pos_tagger,
                                            self.common_lstm,
                                            vocab,
                                            word_to_indices,
                                            RELATIONS_TO_INDICES,
                                            UNIVERSAL_DEPENDENCY_LABELS)

    def forward(self, batch, train=False):

        WORDS, POS, HEADS = 0, 1, 2
        word_level_batch = batch[0]

        character_level_batch = batch[1]

        holistic_embeddings = self.holistic_embedding(word_level_batch[:, WORDS, :])
        pretrained_embeddings = self.pretrained_embedding_layer(word_level_batch[:, WORDS, :])
        character_level_embeddings = self.character_level_model(character_level_batch)
        # ideally [batch_size x num_words x embedding_dim]
        # all three have the same dimensions (ideally)
        # and represent three different representations of the same thing
        final_embeddings = torch.add(torch.add(holistic_embeddings, pretrained_embeddings), character_level_embeddings)

        pos_embeddings = None
        heads_indices = None

        if train:
            pos_embeddings = self.pos_embeddings(word_level_batch[:, 1, :])
            heads_indices = word_level_batch[:, HEADS, :]

        else:
            pos_distributions = self.pos_tagger(word_level_batch[:, 0, :])
            pos_predictions = self.pos_tagger.predict(pos_distributions)
            pos_embeddings = self.pos_embeddings(pos_predictions)

            heads_indices = self.edge_scorer.predict(word_level_batch[:, WORDS, :])

        lstm_inputs = torch.cat((final_embeddings, pos_embeddings), dim=2)

        lstm_outputs, (last_hidden_state, last_cell_state) = self.common_lstm(lstm_inputs)


        edge_scorer = self.edge_scorer.hidden_states_treatment(lstm_outputs, last_hidden_state, last_cell_state)
        edge_labels = self.edge_labeller.hidden_states_treatment(lstm_outputs, last_hidden_state, last_cell_state, heads_indices)

        return (edge_scorer, edge_labels)




def train_epoch(model, optimizer, loss_fun, dataloader):

    for batch in dataloader:
        edge_scorer_results, edge_label_results = model(batch, train=True)

        edge_scorer_results_swapped = torch.swapaxes(edge_scorer_results, 1, 2)
        edge_label_results_swapped = torch.swapaxes(edge_label_results, 1, 2)

        word_level_batch = batch[0]
        target_edge_scorer = word_level_batch[:, 2, :]
        target_edge_labeller = word_level_batch[:, 3, :]
        loss_scorer = loss_fun(edge_scorer_results_swapped, target_edge_scorer)
        loss_labeller = loss_fun(edge_label_results_swapped, target_edge_labeller)

        total_loss = loss_scorer + loss_labeller
        ic(total_loss)
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def train(model, optimizer, loss_fun, dataset, num_epochs):

    for epoch in range(num_epochs):
        print(f"{epoch+1}")
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=69)
        train_epoch(model, optimizer, loss_fun, dataloader)
        print("-------")
        torch.save(model, GRAND_MODEL_PATH)

def train_model(train_path, num_epochs):
    train_dataset = Dataset(from_vocab=False, file_path=train_path, make_character_dataset=True)
    train_dataset.load_pretrained_embeddings(PRETRAINED_EMBEDDING_FILE)

    pos_tagger = torch.load(POS_MODEL_PATH)

    model = Parser(400, 100,
                    len(train_dataset.vocab), 100,
                    200, 100, 17,
                    pos_tagger,
                    100,
                    train_dataset.pretrained_embedding_weights,
                    len(train_dataset.character_dataset.character_vocab),
                    train_dataset.character_dataset.character_vocab,
                    train_dataset.character_dataset.character_to_indices,
                    train_dataset.vocab,
                    train_dataset.words_to_indices).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss_fun = torch.nn.CrossEntropyLoss()

    train(model, optimizer, loss_fun, train_dataset, num_epochs)


train_path = '../data/UD_English-Atis/en_atis-ud-train.conllu'
train_model(train_path, 10)

