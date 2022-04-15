import torch
from icecream import ic

class Dataset(torch.utils.data.Dataset):
  def __init__(self, trainfile=None):
    self.trainfile = trainfile
    self.vocab = []
    self.freqs = {}
    self.words_to_indices = {}
    self.dataset = []

    self.tags_to_indices = {tag: index for index, tag in enumerate(
                              ['ADJ',   'ADP', 'ADV',  'AUX',  'CCONJ', 'DET', 'INTJ',
                               'NOUN',  'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT',
                               'SCONJ', 'SYM', 'VERB', 'X'])}

    dataset = []
    words = []
    postags = []
    heads = []

    freqs = {}
    with open(self.trainfile, "r") as f:
      lines = f.readlines()

      for line in lines:
        # Comment line
        if (len(line) > 0 and line[0] == '#'):
          continue

        line = line.strip('\n')
        # Non-empty line
        if (len(line) > 0):
          columns = line.split('\t')

          WORD_INDEX, POS_INDEX, HEAD_INDEX = 1, 3, 6
          words.append(columns[WORD_INDEX])
          postags.append(columns[POS_INDEX])
          heads.append(int(columns[HEAD_INDEX]))

          try: freqs[columns[1]] += 1
          except KeyError: freqs[columns[1]] = 1

        # Empty line implies that a sentence is finished
        else:
          dataset.append((words, postags, heads))
          self.vocab += words
          words = []
          postags = []
          heads = []

    self.vocab = list(set(self.vocab))
    self.vocab = list(filter(lambda x: freqs[x] > 0, self.vocab))
    self.vocab.append('<UNK>')
    self.freqs = freqs
    self.words_to_indices = {word: index for index, word in enumerate(self.vocab)}

    for (words, tags, heads) in dataset:
      word_indices = [self.index(word) for word in words]
      tag_indices = [self.tags_to_indices[tag] for tag in tags]
      heads_indices = [word_indices[i-1] if (i > 0) else -1 for i in heads]

      self.dataset.append((word_indices, tag_indices, heads_indices))

  def index(self, word):
    try: idx = self.words_to_indices[word]
    except KeyError: idx = len(self.vocab) - 1
    return idx

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index):
    return self.dataset[index]
