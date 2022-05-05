# Dependency
A project for the Introduction to NLP course at IIIT Hyderabad.

# Submission
The locations of the various deliverables are as follows:

* presentation file: [here](https://docs.google.com/presentation/d/196A3EmL12rHvy0UY90hPPvcuXjfh3QntaaZLwxdjTQQ/edit?usp=sharing)
* project report: `report.pdf` in the home directory.
* code: the `src/` subdirectory
* data: the `data/` subdirectory in [this drive folder](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/shashwat_s_research_iiit_ac_in/EpWVVeDMvs5Hqsenxe84_9EBIYz-9rdh5B8cf50FKzGBvQ?e=vAXGl4)
* model checkpoints: the `models/` subdirectory in [this drive folder](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/shashwat_s_research_iiit_ac_in/EpWVVeDMvs5Hqsenxe84_9EBIYz-9rdh5B8cf50FKzGBvQ?e=vAXGl4)
* README: `README.md` in the home directory.

# Reference
Dependency parsing is a popular grammar formalism used for better understanding a sentence structure. A dependency parsing mechanism can give a parsed dependency tree for a given sentence, which involves a set of relations over its words such that one is a ‘dependent’ of the other. Training such dependency parsers involves making use of annotated data to learn the way these trees are constructed, which can be modeled to give good performance by effectively using neural networks. Based on the approach taken to model the problem, the parsing can be either transition-based or graph-based. In this project, the students will implement a graph-based neural model which will be trained to perform dependency parsing on a sentence.  

The architecture of the model is according to [Stanford’s Graph-based Neural Dependency Parser at the CoNLL 2017 Shared Task](https://aclanthology.org/K17-3002/).

The model consists primarily of a POS tagger, an edge scorer, and an edge labeller. The POS tagger is trained separately and the edge scorer and edge labeller are trained by optimising the sum of their losses.

# Running
Clone the repo, change into the `src` directory,
```
> git clone git@github.com:sentient-bread/Dependency.git
> cd src
```
run the `main.py` file and enter the sentence you wish to parse (in lowercase, without punctuation):
```
> python3 main.py
Enter a sentence: the flight goes from houston to atlanta
IDX	WORD	POS	HEAD	RELN
1	the	DET	2	det
2	flight	NOUN	3	nsubj
3	goes	VERB	0	root
4	from	ADP	5	case
5	houston	PROPN	3	obl
6	to	ADP	7	case
7	atlanta	PROPN	3	obl
```

The output is in a format similar to the CoNLL dataset.
