import torch
from icecream import ic
from settings import *

PAD_DEPENDENCY = -1
# the dependency set will use -1 tokens as a pad
# The reason is that -1 refers to the dummy head


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
                               'SCONJ', 'SYM', 'VERB', 'X', 'NULL'])}

    self.length_longest_sequence = 0
    # to store the length of the longest sequence

    dataset = []
    words = []
    postags = []
    heads = []

    freqs = {}
    with open(self.trainfile, "r") as f:
      lines = f.readlines()

      for line in lines:

        len_last_read = len(words)
        if (len_last_read > self.length_longest_sequence):
          self.length_longest_sequence = len_last_read

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
          words.insert(0, '<BOS>')
          postags.insert(0, 'NULL')
          heads.insert(0, -1)
          # This -1 does NOT mean that the head of <BOS> is the last word
          # in the sentence. It means that the head of <BOS> is itself.

          dataset.append((words, postags, heads))
          self.vocab += words
          words = []
          postags = []
          heads = []

    metatokens = [ '<BOS>', '<PAD>', '<UNK>' ]
    for tok in metatokens:
      freqs.update({tok: 1})

    self.vocab = list(set(self.vocab))
    self.vocab = list(filter(lambda x: freqs[x] > 0, self.vocab))
    self.vocab = sorted(self.vocab)

    self.vocab += metatokens
    # last three indices are for <BOS>, <PAD> and <UNK> respectively

    self.freqs = freqs
    self.words_to_indices = {word: index for index, word in enumerate(self.vocab)}

    for (words, tags, heads) in dataset:
      word_indices = [self.index(word) for word in words]
      tag_indices = [self.tags_to_indices[tag] for tag in tags]
      # converts sentence indices to vocabulary indices
      heads_indices = [word_indices[i] if i != -1 else -1 for i in heads]

      self.dataset.append((word_indices, tag_indices, heads_indices))

  def index(self, word):
    try: idx = self.words_to_indices[word]
    except KeyError: idx = self.words_to_indices['<UNK>']
    return idx

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index):
    #ic((torch.tensor(self.dataset[index])).shape)
    to_ret = self.dataset[index]
    seq_length = len(to_ret[0])
    amount_to_pad = self.length_longest_sequence - seq_length
    to_ret_tensor = torch.tensor(to_ret).to(DEVICE)
    to_ret_padded = torch.nn.functional.pad(to_ret_tensor, (0, amount_to_pad), "constant", 0)

    to_ret_padded[0] = torch.nn.functional.pad(
                                  to_ret_tensor[0],
                                  (0, amount_to_pad),
                                  "constant",
                                  len(self.vocab)-2)

    to_ret_padded[1] = torch.nn.functional.pad(to_ret_tensor[1],
                                              (0, amount_to_pad),
                                              "constant",
                                              len(self.tags_to_indices)-1
                                              )

    to_ret_padded[2] = torch.nn.functional.pad(to_ret_tensor[2],
                                              (0, amount_to_pad),
                                              "constant",
                                              PAD_DEPENDENCY,
    )

    return to_ret_padded

# test_dataset = Dataset(trainfile='../data/UD_English-Atis/en_atis-ud-test.conllu')
# test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE)

# for batch in test_dataloader:

#   ic(batch[:, 0, :], batch[:, 0, :].shape)
