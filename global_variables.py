import torch

EMB_DIM = 100

POS_DICT = {"ADJ": 0,
            "ADV": 1,
            "INTJ": 2,
            "NOUN": 3,
            "PROPN": 4,
            "VERB": 5,
            "ADP": 6,
            "AUX": 7,
            "CCONJ": 8,
            "DET": 9,
            "NUM": 10,
            "PART": 11,
            "PRON": 12,
            "SCONJ": 13,
            "PUNCT": 14,
            "SYM": 15,
            "X": 16}

POS_LIST = ["ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB",
            "ADP", "AUX", "CCONJ", "DET", "NUM", "PART", "PRON", "SCONJ",
            "PUNCT", "SYM", "X"]

NB_POS = len(POS_DICT)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

train_mode = "dev"
test_mode = "test"

epochs = 1  # epochs = 0 means no training
batch_size = 100


multilingual = False

SAVE_NET_PATH = "Models/XLMEmbMultiLayerBiLSTMPosTagger_1.model"
LOAD_NET_PATH = "Models/XLMEmbMultiLayerBiLSTMPosTagger_1.model"

verbose = False
