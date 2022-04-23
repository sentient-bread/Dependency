#DEVICE = "cuda:0"
DEVICE = "cpu"
BATCH_SIZE = 100
POS_MODEL_PATH = "../models/pos_model.pth"
EDGESCORER_MODEL_PATH = "../models/edgescorer_model.pth"


# universal dependency labels
UNIVERSAL_DEPENDENCY_LABELS = [


    "root",
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
    "vocative",
    "xcomp",

]

RELATIONS_TO_INDICES = {
    tag: index for index, tag in enumerate(
        UNIVERSAL_DEPENDENCY_LABELS
    )
}
