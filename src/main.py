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

model = torch.load(GRAND_MODEL_PATH, map_location=torch.device('cpu'))
dataset = Dataset(file_path="../data/UD_English-Atis/en_atis-ud-train.conllu",
                  make_character_dataset=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=42)
character_to_indices = model.character_level_model.character_to_indices

tags_list = ['ADJ',   'ADP', 'ADV',  'AUX',  'CCONJ', 'DET', 'INTJ',                     'NOUN',  'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT',                           'SCONJ', 'SYM', 'VERB', 'X', 'NULL']

sentence = input("Enter a sentence: ")
tokens = sentence.strip().split()
tokens.insert(0,'<BOS>')

mst, labels, tags = create_graph(model, tokens)

print("IDX\tWORD\tPOS\tHEAD\tRELN")
for node in range(1, len(tokens)):
  head = mst[node]
  word = tokens[node]
  tag = tags_list[tags[node]]
  label = UNIVERSAL_DEPENDENCY_LABELS[labels[node]]
  print(f"{node}\t{word}\t{tag}\t{head}\t{label}")

