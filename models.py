import torch.nn as nn
import global_variables as glob
import torch
from preprocessing import sentence_to_word_index
import XLM_pre


class WordEmb(nn.Module):
    def __init__(self, num_embedding, embedding_dim):
        super().__init__()
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(num_embedding, embedding_dim)

    def forward(self, indexes):
        return self.embedding(indexes)


class LinearPosTagger(nn.Module):
    """Takes in a batch of word indexes
    Returns a batch of 1d-tensors (probabilities for each POS tag)
    """
    def __init__(self, dico_size):
        super().__init__()
        self.embedding = WordEmb(2*dico_size, glob.EMB_DIM)
        self.lin1 = nn.Linear(in_features=glob.EMB_DIM, out_features=glob.NB_POS)
        self.sm = nn.Softmax(dim=1)

    def forward(self, word_batch):
        out = self.embedding(word_batch)
        out = self.lin1(out)
        out = self.sm(out)
        return out

    def predict(self, word_batch):
        """returns the maximum probability and the corresponding POS-tag"""
        out = self.forward(word_batch)
        values, indices = out.max(1)
        return values, [glob.POS_LIST[i] for i in indices]


class LSTMPosTagger(nn.Module):
    """Takes in a sequence of word embeddings -> shape (len(sequence) , EMB_DIM)
    Outputs [hidden states, last state] returned by the lstm network
    """
    def __init__(self, dico_size, hidden_dim=100):
        super().__init__()
        self.embedding = WordEmb(2 * dico_size, glob.EMB_DIM)
        self.lstm = nn.LSTM(glob.EMB_DIM, hidden_dim)
        self.lin1 = nn.Linear(in_features=hidden_dim, out_features=glob.NB_POS)

    def forward(self, word_sequence):
        """word_sequence.shape = (len(sentence))"""
        emb = self.embedding(word_sequence)
        hidden, _ = self.lstm(emb.view(len(word_sequence), 1, -1))
        out = self.lin1(hidden)
        return out.squeeze(1)

    def predict(self, word_sequence):
        """for each word in the sentence, returns the max probability
        and the corresponding POS-tag"""
        out = self.forward(word_sequence).squeeze(1)
        values, indices = out.max(1)
        return values, [glob.POS_LIST[i] for i in indices]


class BiLSTMPosTagger(nn.Module):
    """Takes in a sequence of words embeddings
    feeds it in to LSTMs in parallel (one in reverse)
    """
    def __init__(self, dico_size, hidden_dim=100):
        super().__init__()
        self.embedding = WordEmb(2 * dico_size, glob.EMB_DIM)
        self.lstm_fwd = nn.LSTM(glob.EMB_DIM, hidden_dim)
        self.lstm_bwd = nn.LSTM(glob.EMB_DIM, hidden_dim)
        self.lin1 = nn.Linear(in_features=2*hidden_dim, out_features=glob.NB_POS)

    def forward(self, word_sequence):
        emb = self.embedding(word_sequence)
        h_fwd, _ = self.lstm_fwd(emb.unsqueeze(1))
        h_bwd, _ = self.lstm_bwd(emb.unsqueeze(1).flip(0))
        bi_h = torch.cat((h_fwd, h_bwd), dim=2)
        out = self.lin1(bi_h)
        return out.squeeze(1)

    def predict(self, word_sequence):
        """for each word in the sentence, returns the max probability
        and the corresponding POS-tag"""
        out = self.forward(word_sequence).squeeze(1)
        values, indices = out.max(1)
        return values, [glob.POS_LIST[i] for i in indices]


class BiLSTMPosTaggerEXPERIMENT(nn.Module):
    """Takes in a sequence of words embeddings
    feeds it in to a forward LSTM -> h
    feeds h to a backward LSTM
    """
    def __init__(self, dico_size, hidden_dim):
        super().__init__()
        self.embedding = WordEmb(2 * dico_size, glob.EMB_DIM)
        self.lstm_fwd = nn.LSTM(glob.EMB_DIM, hidden_dim)
        self.lstm_bwd = nn.LSTM(hidden_dim, hidden_dim)
        self.lin1 = nn.Linear(in_features=hidden_dim, out_features=glob.NB_POS)

    def forward(self, word_sequence):
        emb = self.embedding(word_sequence)
        h_fwd, _ = self.lstm_fwd(emb.unsqueeze(1))
        h_bwd, _ = self.lstm_bwd(h_fwd.flip(0))
        out = self.lin1(h_bwd)
        return out.squeeze(1)

    def predict(self, word_sequence):
        """for each word in the sentence, returns the max probability
        and the corresponding POS-tag"""
        out = self.forward(word_sequence).squeeze(1)
        values, indices = out.max(1)
        return values, [glob.POS_LIST[i] for i in indices]


class BiLSTM(nn.Module):
    """Takes in a sequence
    feeds it in to LSTMs in parallel (one in reverse)
    returns the sequence of concatenations of the hidden layers
    """
    def __init__(self, input_dim, hidden_dim=100):
        super().__init__()
        self.emb_dim = glob.EMB_DIM
        self.hidden_dim = hidden_dim
        self.lstm_fwd = nn.LSTM(input_dim, hidden_dim)
        self.lstm_bwd = nn.LSTM(input_dim, hidden_dim)

    def forward(self, word_sequence):
        h_fwd, _ = self.lstm_fwd(word_sequence)
        h_bwd, _ = self.lstm_bwd(word_sequence.flip(0))
        bi_h = torch.cat((h_fwd, h_bwd), dim=2)
        return bi_h


class MultiLayerBiLSTMPosTaggerHOMEMADE(nn.Module):
    """Takes in a sequence of words embeddings
    feeds it to two layers of BiLSTMs"""
    def __init__(self, dico_size, hidden_dim=100):
        super().__init__()
        self.embedding = WordEmb(2 * dico_size, glob.EMB_DIM)
        self.bi1 = BiLSTM(input_dim=glob.EMB_DIM, hidden_dim=hidden_dim)
        self.bi2 = BiLSTM(input_dim=2*hidden_dim, hidden_dim=hidden_dim)
        self.lin1 = nn.Linear(in_features=2*hidden_dim, out_features=glob.NB_POS)

    def forward(self, word_sequence):
        emb = self.embedding(word_sequence)
        h_1 = self.bi1(emb.unsqueeze(1))
        h_2 = self.bi2(h_1)
        out = self.lin1(h_2)
        return out.squeeze(1)

    def predict(self, word_sequence):
        """for each word in the sentence, returns the max probability
        and the corresponding POS-tag"""
        out = self.forward(word_sequence).squeeze(1)
        values, indices = out.max(1)
        return values, [glob.POS_LIST[i] for i in indices]


class MultiLayerBiLSTMPosTagger(nn.Module):
    """Takes in a sequence of words embeddings
    feeds it to two layers of BiLSTMs"""
    def __init__(self, dico_size, hidden_dim=100):
        super().__init__()
        self.embedding = WordEmb(2 * dico_size, glob.EMB_DIM)
        self.BiLSTMs = nn.LSTM(input_size=glob.EMB_DIM, hidden_size=hidden_dim,
                               bidirectional=True, num_layers=2)
        self.lin1 = nn.Linear(in_features=2*hidden_dim, out_features=glob.NB_POS)

    def forward(self, word_sequence):
        emb = self.embedding(word_sequence)
        h, _ = self.BiLSTMs(emb.unsqueeze(1))
        out = self.lin1(h)
        return out.squeeze(1)

    def predict(self, word_sequence):
        """for each word in the sentence, returns the max probability
        and the corresponding POS-tag"""
        out = self.forward(word_sequence).squeeze(1)
        values, indices = out.max(1)
        return values, [glob.POS_LIST[i] for i in indices]


class XLMEmbMultiLayerBiLSTMPosTagger(nn.Module):
    """Takes in a sequence of words indices
    feeds it to two layers of BiLSTMs"""
    def __init__(self, dico, hidden_dim=100):
        super().__init__()
        self.dico = dico
        self.embedding = WordEmb(2 * len(dico), glob.EMB_DIM)
        self.BiLSTMs = nn.LSTM(input_size=glob.EMB_DIM + 1024, hidden_size=hidden_dim,
                               bidirectional=True, num_layers=2)
        self.lin1 = nn.Linear(in_features=2*hidden_dim, out_features=glob.NB_POS)

    def forward(self, sentence):
        xlm_batch = XLM_pre.create_batch([sentence])[1:-1]
        xlm_batch = xlm_batch.detach()           # do not train the pretrained XLM weights

        indexes = sentence_to_word_index(self.dico, sentence)
        emb = self.embedding(indexes).unsqueeze(1)

        emb_concat = torch.cat((xlm_batch, emb), 2)

        h, _ = self.BiLSTMs(emb_concat)
        out = self.lin1(h)
        return out.squeeze(1)

    def predict(self, word_sequence):
        """for each word in the sentence, returns the max probability
        and the corresponding POS-tag"""
        out = self.forward(word_sequence).squeeze(1)
        values, indices = out.max(1)
        return values, [glob.POS_LIST[i] for i in indices]
