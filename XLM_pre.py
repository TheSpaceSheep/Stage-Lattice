import torch
import global_variables as glob
from XLM.src.utils import AttrDict
from XLM.src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from XLM.src.model.transformer import TransformerModel

"""_________________________________________ XML ________________________________________________"""

model_path = 'pre-trained_embeddings/mlm_enfr_1024.pth'
reloaded = torch.load(model_path)
params = AttrDict(reloaded['params'])

# build dictionary / update parameters
dicoXLM = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
params.n_words = len(dicoXLM)
params.bos_index = dicoXLM.index(BOS_WORD)
params.eos_index = dicoXLM.index(EOS_WORD)
params.pad_index = dicoXLM.index(PAD_WORD)
params.unk_index = dicoXLM.index(UNK_WORD)
params.mask_index = dicoXLM.index(MASK_WORD)

# build model / reload weights
XLMmodel = TransformerModel(params, dicoXLM, True, True)
XLMmodel.load_state_dict(reloaded['model'])


def sen_list_to_xlm_sen_list(sentences):
    sen_list = []
    for s in sentences:
        sen_list.append((s.as_str(), 'fr'))
    return sen_list


def create_batch(sentences):
    """returns the XLM embeddings of words in sentences of the list 'sentences'
    shape : (max_sent_length, batch_size, 1024)"""
    sentences_xlm = sen_list_to_xlm_sen_list(sentences)
    sentences_xlm = [(('</s>\t%s\t</s>' % sent.strip()).split('\t'), lang) for sent, lang in sentences_xlm]
    bs = len(sentences_xlm)
    slen = max([len(sent) for sent, _ in sentences_xlm])
    word_ids = torch.LongTensor(slen, bs).fill_(params.pad_index)
    for i in range(len(sentences_xlm)):
        sent = torch.LongTensor([dicoXLM.index(w) for w in sentences_xlm[i][0]])
        word_ids[:len(sent), i] = sent
    lengths = torch.LongTensor([len(sent) for sent, _ in sentences_xlm])
    langs = torch.LongTensor([params.lang2id[lang] for _, lang in sentences_xlm]).unsqueeze(0).expand(slen, bs)
    out = XLMmodel('fwd', x=word_ids, lengths=lengths, langs=langs, causal=False).contiguous().to(glob.device)
    out.to(glob.device)

    return out
