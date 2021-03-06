# DEVICE = "cuda:0"
DEVICE = "cpu"
BATCH_SIZE = 150
POS_MODEL_PATH = "../models/pos_model.pth"
EDGESCORER_MODEL_PATH = "../models/edgescorer_model.pth"
CHARACTER_MODEL_PATH = "../models/character_model.pth"
EDGELABELLER_MODEL_PATH = "../models/edgelabeller_model.pth"

GRAND_MODEL_PATH = "../models/grand_model.pth"
GRAND_MODEL_PATH_HINDI = "../models/grand_model_hindi.pth"
GRAND_MODEL_PATH_SANSKRIT = "../models/grand_model_sanskrit.pth"

PRETRAINED_EMBEDDING_FILE = "../embeddings/glove.6B.100d.txt"



POS_MODEL_PATH_HINDI = "../models/pos_hindi.pth"
POS_MODEL_PATH_SANSKRIT = "../models/pos_sanskrit.pth"


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

RELATIONS_TO_INDICES = {
    tag: index for index, tag in enumerate(
        UNIVERSAL_DEPENDENCY_LABELS
    )
}
