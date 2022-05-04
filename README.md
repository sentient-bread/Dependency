# Dependency
A project for the Introduction to NLP course at IIIT Hyderabad.

# Reference
Dependency parsing is a popular grammar formalism used for better understanding a sentence structure. A dependency parsing mechanism can give a parsed dependency tree for a given sentence, which involves a set of relations over its words such that one is a ‘dependent’ of the other. Training such dependency parsers involves making use of annotated data to learn the way these trees are constructed, which can be modeled to give good performance by effectively using neural networks. Based on the approach taken to model the problem, the parsing can be either transition-based or graph-based. In this project, the students will implement a graph-based neural model which will be trained to perform dependency parsing on a sentence.  

The architecture of the model is according to [Stanford’s Graph-based Neural Dependency Parser at the CoNLL 2017 Shared Task](https://aclanthology.org/K17-3002/).

The model consists primarily of a POS tagger, an edge scorer, and an edge labeller. The POS tagger is trained separately and the edge scorer and edge labeller are trained by optimising the sum of their losses.
