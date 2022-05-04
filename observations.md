* POS tagger (trained on 20 epochs) has consistently low recall for adverbs. Maybe because of low data + lack of lexical cues? Because low data is there for interjections and numerals also, but those can be determined lexically.

* Edgescorer (35 epochs) has 91.3% attachment score on dev and 92.7% on test (with simple argmax). Using the MST generator increases both these by ~0.2%. (TODO: error analysis)
* With argmax and num_layers = 2, 91.4% and 92.3%
* num_layers = 3, 91.5% and 92.8%

* POS tagger achieved 98% after 20 epochs of training on English (Atis). On Chinese (GSD), it achieved 84.3% after 20 epochs and took 30 epochs to stabilise around 85%.
* Edge scorer took 8 epochs to stabilise around 58.5%.

* Edgescorer with faulty POS:
| POS Accuracy | ES Max Accuracy | Num_Epochs     |
| 15.2%        | 90.8%           | 11 (89.9) - 20 |
| 52.7%        | 90.2%           | 13 (89.5) - 24 |
| 81.1%        | 89.6%           | 13 (89.5) - 24 |
| 86.8%        | 90.4%           | 7  (89.1) - 16 |
| 92%          | 90.3%           | 6  (89.9) - 11 |
| 96.5%        | 90.9%           | 8  (90.4) - 14 |

* ES trained on 13.2% accurate POS tagger achieves 91.5% accuracy. Replaced with 98.6% accurate POS tagger, achieves 68.1% accuracy. It used bad tags?
* ES without POS tagger achieves 90.9% accuracy. It's not needed?

# Notes about the character model

- Each word is represented as a list of characters and is padded to the right for now 
- the lstm of the character model goes word by word. We feel that this is a problem for examples in hindi like empty verbs: "naach rahe hai" we don't want to separate the context of "rahe hai" from "naach". But the language tokenizes this morphemic information and this could have been covered by the character level model.

* We're training one LSTM **each** for the EdgeScorer and EdgeLabeller classes,
  and getting our hidden states that we pass to MLPs internally to each class.
* This may cause issues because the hidden states that are inputs to each of
  the four MLPs may not all be the same now. The two for the EdgeScorer and the
  two for the EdgeLabeller are of course the same for each other, but all four
  need not be the same.
* For now, we're assuming that it doesn't matter that the hidden states are not
  the exact same since these biaffine classifiers are being trained in
  isolation any way.

# POS Tagging training report

- when trained with a hidden size of 50 (therefore 100 due to biLSTM) the model performed with metrics that were better than hidden size 200

- But the paper prescribes 200 (therefore 400), that is what we are currently going for

Results when trained with hidden size of 50 (100)

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


Results when trained with hidden size of 200 (400)

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

# POS Tagger for Hindi

Overall
              precision    recall  f1-score   support

         ADJ       0.92      0.86      0.89      2043
         ADP       0.99      1.00      0.99      7544
         ADV       0.89      0.86      0.87       304
         AUX       0.97      0.98      0.97      2596
       CCONJ       0.98      1.00      0.99       635
         DET       0.95      0.96      0.96       745
        NOUN       0.91      0.91      0.91      8036
        NULL       1.00      1.00      1.00     77398
         NUM       0.96      0.85      0.90       693
        PART       0.99      0.96      0.98       677
        PRON       0.98      0.97      0.97      1372
       PROPN       0.84      0.88      0.86      4438
       PUNCT       1.00      1.00      1.00      2420
       SCONJ       0.98      0.99      0.99       655
        VERB       0.96      0.94      0.95      3263
           X       0.21      0.33      0.26         9

    accuracy                           0.98    112828
   macro avg       0.91      0.90      0.91    112828
weighted avg       0.98      0.98      0.98    112828

# POS Tagger for Sanskrit

Overall
              precision    recall  f1-score   support

         ADJ       0.44      0.33      0.37       870
         ADV       0.92      0.87      0.90      1084
         AUX       0.80      0.44      0.57        90
       CCONJ       0.99      0.96      0.98       152
         DET       0.88      0.15      0.25        48
        NOUN       0.77      0.74      0.75      3074
        NULL       1.00      1.00      1.00    226008
         NUM       0.86      0.28      0.42        89
        PART       0.99      0.99      0.99       785
        PRON       0.92      0.88      0.90      1443
       SCONJ       0.86      0.82      0.84        97
        VERB       0.62      0.82      0.71      1940

    accuracy                           0.99    235680
   macro avg       0.84      0.69      0.72    235680
weighted avg       0.99      0.99      0.99    235680



# Attachment scores

Attachment label: 0.9492089925062448
Attachment heads: 0.9685863874345549


# Hindi attachment scores

Attachment label: 0.9090909090909091
Attachment heads: 0.9162303664921466

# Sanskrit attachment scores

Attachment label: 0.4508990318118949
Attachment heads: 0.5780141843971631




