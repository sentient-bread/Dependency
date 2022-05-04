import torch
from torch import nn
from icecream import ic
from settings import *
from character_model import CharacterModel
from edgescorer import EdgeScorer
from edgelabeller import EdgeLabeller
from final_model import Parser
from data import Dataset
from graph import create_graph

model = torch.load(GRAND_MODEL_PATH)
dataset = Dataset(file_path="../data/UD_English-Atis/en_atis-ud-train.conllu",
                  make_character_dataset=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=42)
character_to_indices = model.character_level_model.character_to_indices

ic(next(iter(dataloader))[0].shape)
ic(next(iter(dataloader))[1].shape)

sentence = input("Enter a sentence: ")
tokens = sentence.strip().split()

indices = torch.tensor([model.index(tok) for tok in tokens]).to(DEVICE)
pos = torch.tensor([0 for _ in tokens]).to(DEVICE)
heads = torch.tensor([0 for _ in tokens]).to(DEVICE)
labels = torch.tensor([0 for _ in tokens]).to(DEVICE)

word_level = torch.stack([indices, pos, heads, labels], dim=0).unsqueeze(0)
# unsqueeze for batch

max_word_len = max([len(tok) for tok in tokens])
char_level = torch.tensor(model.character_level_model.dataset.word_index_list_to_character_list(
        indices.tolist(), max_word_len
        )).to(DEVICE)

ic(char_level)
ic(char_level.shape)
