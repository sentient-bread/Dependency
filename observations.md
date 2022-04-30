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

* We're training one LSTM **each** for the EdgeScorer and EdgeLabeller classes,
  and getting our hidden states that we pass to MLPs internally to each class.
* This may cause issues because the hidden states that are inputs to each of
  the four MLPs may not all be the same now. The two for the EdgeScorer and the
  two for the EdgeLabeller are of course the same for each other, but all four
  need not be the same.
* For now, we're assuming that it doesn't matter that the hidden states are not
  the exact same since these biaffine classifiers are being trained in
  isolation any way.
