* POS tagger (trained on 20 epochs) has consistently low recall for adverbs. Maybe because of low data + lack of lexical cues? Because low data is there for interjections and numerals also, but those can be determined lexically.
* Edgescorer (35 epochs) has 91.3% attachment score on dev and 92.7% on test (with simple argmax). Using the MST generator increases both these by ~0.2%. (TODO: error analysis)
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

* TODO: train ES on bad POS tagger and run it with a good POS tagger. Does it ignore or use fake POS?
