import torch
from icecream import ic
from settings import *
<<<<<<< HEAD
import random
=======
>>>>>>> origin/main
import numpy

PAD_DEPENDENCY = 0
# the dependency set will use -1 tokens as a pad
# The reason is that -1 refers to the dummy head

class Dataset(torch.utils.data.Dataset):
  def __init__(self, from_vocab=False, file_path=None, vocab=None, words_to_indices=None, make_character_dataset=False):
    # cheap constructor overloading

    self.tags_to_indices = {tag: index for index, tag in enumerate(
                              ['ADJ',   'ADP', 'ADV',  'AUX',  'CCONJ', 'DET', 'INTJ',
                               'NOUN',  'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT',
                               'SCONJ', 'SYM', 'VERB', 'X', 'NULL'])}

    if from_vocab:
      self.vocab = vocab
      self.words_to_indices = words_to_indices

    else:
      self.vocab = []
      self.freqs = {}
      self.words_to_indices = {}


    self.file_path = file_path
    self.dataset = []

    dataset = []
    words = []
    postags = []
    heads = []
    relations = []

    self.length_longest_sequence = 0

    freqs = {} # relevant only if from_vocab=True

    with open(self.file_path, "r") as f:
      lines = f.readlines()

      for line in lines:
        len_last_read = len(words)

        if len_last_read > self.length_longest_sequence:
          self.length_longest_sequence = len_last_read

        # Comment line
        if (len(line) > 0 and line[0] == '#'):
          continue

        line = line.strip('\n')
        # Non-empty line
        if (len(line) > 0):
          columns = line.split('\t')

          WORD_INDEX, POS_INDEX, HEAD_INDEX, RELATION_INDEX = 1, 3, 6, 7
          words.append(columns[WORD_INDEX])
          postags.append(columns[POS_INDEX])
          heads.append(int(columns[HEAD_INDEX]))
          relations.append(columns[RELATION_INDEX])

          if not from_vocab:
            try: freqs[columns[1]] += 1
            except KeyError: freqs[columns[1]] = 1

        # Empty line implies that a sentence is finished
        else:
          words.insert(0, '<BOS>')
          postags.insert(0, 'NULL')
          heads.insert(0, 0)
          relations.insert(0, "<null>")
          # This -1 does NOT mean that the head of <BOS> is the last word
          # in the sentence. It means that the head of <BOS> is itself.

          dataset.append((words, postags, heads, relations))
          if not from_vocab:
            self.vocab += words
          words = []
          postags = []
          heads = []
          relations = []

    self.length_longest_sequence += 1

    if not from_vocab:
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

    for (words, tags, heads, relations) in dataset:
      word_indices = [self.index(word) for word in words]
      tag_indices = [self.tags_to_indices[tag] for tag in tags]
      relation_indices = [RELATIONS_TO_INDICES[relation] for relation in relations]
      # converts sentence indices to vocabulary indices

      # heads already contains correct indices, because when <BOS>
      # was added, 1-based indexing became 0-based, as needed.
      self.dataset.append((word_indices, tag_indices, heads, relation_indices))

    self.make_character_dataset = make_character_dataset
    self.character_dataset = DatasetCharacter(word_vocab=self.vocab,
                                              word_to_indices=self.words_to_indices,
                                              dataset_array=self.dataset,
                                              length_longest_sentence=self.length_longest_sequence)

  def load_pretrained_embeddings(self, embedding_file_path):

    """
    Loads all embeddings from a file
    for all words in the vocabulary, it gets the embedding and puts them in a tensor
    whenever there is a word in the vocabulary that isn't in the embeddings on file
    we just put all 0's
    """
    words = []
    words_to_indices = {}

    embedding_list = []

    index = 0

    # currently assumed that word embeddings are 100 dimensional
    with open(embedding_file_path, "rb") as embedding_file:
      for l in embedding_file:
        line = l.decode().split()
        word = line[0]
        words.append(word)

        words_to_indices[word] = index
        index += 1

        vector = numpy.array(line[1:]).astype(numpy.float)
        embedding_list.append(vector)


    weights_matrix = torch.zeros([len(self.vocab), 100])

    for i, word in enumerate(self.vocab):
      try:
        weights_matrix[i] = torch.tensor(embedding_list[words_to_indices[word]])
        # only the embedding_list[] access can give key error
        # because we are iterating over the vocab
      except KeyError:
        weights_matrix[i] = torch.zeros(100)

    self.pretrained_embedding_weights = weights_matrix

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
    # above padding was done just to change dimensionality

    index_of_sentence = 0
    index_of_pos_tags = 1
    index_of_tree_info = 2
    index_of_relations = 3

    to_ret_padded[index_of_sentence] = torch.nn.functional.pad(
                                  to_ret_tensor[index_of_sentence],
                                  (0, amount_to_pad),
                                  "constant",
                                  len(self.vocab)-2)

    to_ret_padded[index_of_pos_tags] = torch.nn.functional.pad(to_ret_tensor[index_of_pos_tags],
                                              (0, amount_to_pad),
                                              "constant",
                                              len(self.tags_to_indices)-1
                                              )

    to_ret_padded[index_of_tree_info] = torch.nn.functional.pad(to_ret_tensor[index_of_tree_info],
                                              (0, amount_to_pad),
                                              "constant",
                                              PAD_DEPENDENCY,
    )

    to_ret_padded[index_of_relations] = torch.nn.functional.pad(to_ret_tensor[index_of_relations],
                                              (0, amount_to_pad),
                                              "constant",
                                              RELATIONS_TO_INDICES["<null>"],
    )
    if self.make_character_dataset:
      character_tensor = self.character_dataset.__getitem__(index)
      return to_ret_padded, character_tensor

    return to_ret_padded


class DatasetCharacter(torch.utils.data.Dataset):

  """
  This class is for the specific dataset of the character model
  """

  def __init__(self,
              word_vocab=None,
              word_to_indices=None,
              dataset_array=None,
              length_longest_sentence=0):

    """
    The DatasetCharacter is tied to a Dataset class
    the dataset_array is an array of tuples
    The 0th element of each tuple consists of the indices of the list of tokens in the sentence
    """
    super().__init__()
    self.word_vocab = word_vocab
    self.words_to_indices=  word_to_indices
    self.dataset_array = dataset_array
    self.length_longest_sentence = length_longest_sentence




    self.character_to_indices = {}
    self.character_vocab = []
    self.length_longest_word = 0

    self.process_word_vocab(self.word_vocab)

    self.dataset = []
    self.dataset = [self.word_index_list_to_character_list(tup[0],
                                                            self.length_longest_word) for tup in dataset_array]

  def process_word_vocab(self, vocab):
    len_longest_word = 0
    character_set = set()
    # to create character vocab
    for word in vocab:
      if len_longest_word < len(word):
        len_longest_word = len(word)
      for character in word:
        character_set.add(character)

    self.length_longest_word = len_longest_word
    self.character_vocab = sorted(list(character_set), key=ord)
    # sort according to unicode because it's nice
    self.character_vocab.append("<PAD>")
    self.character_vocab.append("<EOS>")

    self.character_to_indices = {character: index for index, character in enumerate(self.character_vocab)}


  def word_to_character_index_list(self, word, pad, left_pad=False):
    """
    replaces each character with index
    and right pads with the padding token index (by default)
    or left pads
    """
    total_length: int = len(word) + pad
    padding_token = len(self.character_vocab) - 2

    if left_pad:
      return [padding_token if i < pad else self.character_to_indices[word[i-pad]]
                for i in range(total_length)]

    return [self.character_to_indices[word[i]] if i < len(word) else padding_token
              for i in range(total_length)]

  def word_index_list_to_character_list(self, indices, len_longest_word, left_pad=False):

    # return [
    #   self.word_to_character_index_list(self.word_vocab[index],
    #                               len_longest_word-len(self.word_vocab[index]), left_pad)
    #   for index in indices

    # ]
    ret_list = []

    for index in indices:
      ret_list.append(self.word_to_character_index_list(self.word_vocab[index],
                                                        len_longest_word-len(self.word_vocab[index]),
                                                        left_pad))
    return ret_list



  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index):
    to_ret = self.dataset[index]
    sentence_length = len(to_ret)
    to_pad_sentence = self.length_longest_sentence - sentence_length

    # Note the <PAD> word token is considered as a sequence of <pad> characters
    # length equal to the longest word
    padding = to_pad_sentence * [ self.length_longest_word * [len(self.character_vocab)-2]]
    to_ret_padded = to_ret + padding
    to_ret_tensor = torch.tensor(to_ret_padded).to(DEVICE)
    # ic(to_ret_tensor.shape) # [sentence_size x word_size]
    return to_ret_tensor


# test_dataset = Dataset(False, file_path='../data/UD_English-Atis/en_atis-ud-test.conllu', character_dataset=True)
# test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=1)

# ic(next(iter(test_dataloader))[1].shape) # batch_size x sentence_size x batch_size



# test_dataset = Dataset(False, file_path='../data/UD_English-Atis/en_atis-ud-test.conllu')
# test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE)
# embedding_file_path = "../embeddings/glove.6B.100d.txt"
# test_dataset.load_pretrained_embeddings(embedding_file_path)
# ic(len(test_dataset.vocab))
# ic(test_dataset.pretrained_embedding_weights.shape)
# for batch in test_dataloader:

#   ic(batch[:, 0, :], batch[:, 0, :].shape)