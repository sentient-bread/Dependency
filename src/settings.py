DEVICE = "cuda:0"
# DEVICE = "cpu"
BATCH_SIZE = 100
POS_MODEL_PATH = "../models/pos_model.pth"
EDGESCORER_MODEL_PATH = "../models/edgescorer_model.pth"
CHARACTER_MODEL_PATH = "../models/character_model.pth"
EDGELABELLER_MODEL_PATH = "../models/edgelabeller_model.pth"

WORDS_IN_BATCH, POS_IN_BATCH, HEADS_IN_BATCH, LABELS_IN_BATCH = range(4)
WORD_LEVEL, CHARACTER_LEVEL = range(2)

UNIVERSAL_DEPENDENCY_LABELS = [
    "<null>", # --
    "acl",
    "acl:relcl",
    "advcl",
    "advmod",
    "advmod:emph",
    "advmod:lmod",
    "amod",
    "appos",
    "aux",
    "aux:pass",
    "case",
    "cc",
    "cc:preconj",
    "ccomp",
    "clf",
    "compound",
    "compound:lvc",
    "compound:prt",
    "compound:redup",
    "compound:svc",
    "conj",
    "cop",
    "csubj",
    "csubj:pass",
    "dep",
    "det",
    "det:numgov",
    "det:nummod",
    "det:poss",
    "det:predet", # --
    "discourse",
    "dislocated",
    "expl",
    "expl:impers",
    "expl:pass",
    "expl:pv",
    "fixed",
    "flat",
    "flat:foreign",
    "flat:name",
    "goeswith",
    "iobj",
    "list",
    "mark",
    "nmod",
    "nmod:poss",
    "nmod:tmod",
    "nsubj",
    "nsubj:pass",
    "nummod",
    "nummod:gov",
    "obj",
    "obl",
    "obl:agent",
    "obl:arg",
    "obl:lmod",
    "obl:tmod",
    "orphan",
    "parataxis",
    "punct",
    "reparandum",
    "root",
    "vocative",
    "xcomp",
]

POS_TAGS = [
    'ADJ',
    'ADP',
    'ADV',
    'AUX',
    'CCONJ',
    'DET',
    'INTJ',
    'NOUN',
    'NUM',
    'PART',
    'PRON',
    'PROPN',
    'PUNCT',
    'SCONJ',
    'SYM',
    'VERB',
    'X',
    'NULL'
]

RELATIONS_TO_INDICES = {
    tag: index for index, tag in enumerate(
        UNIVERSAL_DEPENDENCY_LABELS
    )
}

TAGS_TO_INDICES = {
        tag: index for index, tag in enumerate(
            POS_TAGS
    )
}
