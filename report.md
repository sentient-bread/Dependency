---
title: Introduction to NLP (CS7.401)
subtitle: |
          | Spring 2022, IIIT Hyderabad
          | Project Report
author: |
        | Abhinav S Menon (2020114001)
        | Pratyaksh Gautam (2020114002)
        | Shashwat Singh (2020114016)
---

# Introduction
We have implemented a graph-based dependency parser according to Stanford's submission for the 2017 CoNLL shared task. It uses a POS tagger and a biaffine edge scorer and edge labeller.

## Word Embeddings
The word embeddings are created by summing up three vectors for each word: a pre-trained embedding, an ordinary trained embedding, and a character-level embedding.

The pre-trained embeddings are taken from GloVe's 100-dimensional embeddings.

The trained embeddings are formed from a randomly initialised matrix.

The character-level embedding uses an LSTM and computes attention over the sequence of hidden states, followed by concatenation to the cell state and linear transformation to the word embedding space.

## POS Tagger
The POS tagger uses the above word embeddings passed through an LSTM, with an affine layer applied to the hidden states.

The embeddings for the tags (required for the parser) are extracted from the weight matrix of the affine layer.

## Parser
The parser consists of two biaffine classifiers: an edge scorer and an edge labeller. Both of these take as input the outputs of a BiLSTM, run on the embeddings described above.

The edge scorer takes a sentence and returns, for each word $w_j$, a probability distribution $P(w_i)$ = the probability that $w_i$ is the head of $w_j$. The head of the root word is taken to be the `<BOS>` token.

The edge labeller takes a pair of words (a head and a dependent) and classifies the edge from the head to the dependent into one of the universal dependency relations.

During training, the heads are found by simply taking the highest-probability head for each word. However, this may not always result in a valid parse. For final testing, the MST algorithm described in [Jurafsky & Martin, Chapter 14](https://web.stanford.edu/~jurafsky/slp3/14.pdf) is used, which iteratively identifies and removes cycles.

# Results
## POS Tagger
The results of the UPOS tagger, trained and tested on the English Atis datasets, are as follows.

### Hidden Size of $50 \times 2$
```
Overall
              precision    recall  f1-score   support

         ADJ       0.92      0.97      0.95       220
         ADP       0.98      1.00      0.99      1434
         ADV       0.98      0.71      0.82        76
         AUX       0.99      0.99      0.99       256
       CCONJ       1.00      0.99      1.00       109
         DET       1.00      0.99      0.99       512
        INTJ       1.00      1.00      1.00        36
        NOUN       0.95      0.99      0.97       995
        NULL       1.00      1.00      1.00     13344
         NUM       0.97      0.84      0.90       127
        PART       0.98      0.96      0.97        56
        PRON       0.98      1.00      0.99       392
       PROPN       0.99      0.99      0.99      1738
        VERB       0.99      0.94      0.96       629

    accuracy                           0.99     19924
   macro avg       0.98      0.96      0.97     19924
weighted avg       0.99      0.99      0.99     19924
```

### Hidden Size of $200 \times 2$
(note: this is as prescribed by the paper)
```
Overall
              precision    recall  f1-score   support

         ADJ       0.89      0.96      0.93       220
         ADP       0.99      1.00      0.99      1434
         ADV       0.94      0.76      0.84        76
         AUX       0.99      0.98      0.99       256
       CCONJ       1.00      0.99      1.00       109
         DET       0.99      0.98      0.99       512
        INTJ       0.97      1.00      0.99        36
        NOUN       0.96      0.99      0.97       995
        NULL       1.00      1.00      1.00     13344
         NUM       0.96      0.83      0.89       127
        PART       0.98      0.93      0.95        56
        PRON       0.98      0.99      0.99       392
       PROPN       0.99      0.99      0.99      1738
        VERB       0.99      0.95      0.97       629

    accuracy                           0.99     19924
   macro avg       0.97      0.95      0.96     19924
weighted avg       0.99      0.99      0.99     19924
```

## Parser
The edge scorer achieves an unlabelled attachment score (UAS) of 98.9%.

The edge labeller achieves a labelled attachement score (LAS) of 63.0%.

# Analysis
## POS Tagger
We note from both UPOS taggers that the recall of adverbs and numbers is low. This means that the model was unable to identify a significant fraction of adverbs and numbers.

One factor common to both these parts of speech is the relatively small number of samples available – 76 for adverbs and 127 for numbers (as compared to, say, 995 for nouns and 1434 for adpositions). However, interjections and particles also have comparably low counts.

One possible reason for this might be that interjections and particles can be decided *lexically*. In other words, the model can decide if a word is an interjection or a particle from the word itself, with no knowledge of the context, which makes it easy. For example, `oh` is always an interjection, regardless of the surrounding words.  
On the other hand, adverbs cannot be decided so easily. For example, *good* and *early* may be adverbs or adjectives depending on how they are used. Now the number of samples of adjectives is greater, which may bias the model towards classifying ambiguous cases as adjectives.

As a side effect of this, we expect the precision of adjectives to drop (due to the adverbs wrongly classified as adjectives). This is in fact what we observe, with the precision of adverbs being the next lowest value in the above table.

## Parser
