import torch
import global_variables as glob


class Word:
    """
    w_idx : index of the word in the sentence
    word : the actual string representing the word

    """
    def __init__(self, w_idx, word, pos):
        self.w_idx = w_idx
        self.word = word
        self.pos = pos

    def __repr__(self):
        return "{0}\t{1}\t{2}".format(self.w_idx, self.word, self.pos)


class Sentence:
    def __init__(self, s_idx, words):
        self.s_idx = s_idx
        self.words = words

    def __repr__(self):
        s = ""
        for w in self.words:
            s += w.word + " "

        return "Sentence {} : ".format(self.s_idx) + s

    def as_str(self):
        l = []
        for w in self.words:
            l.append(w.word)
        return "\t".join(l)

    def add_word(self, word):
        self.words.append(word)


# The following are functions that require the above classes
# (put here to avoid circular dependencies)
def read_file(file):
    """Reads the file and returns a list of Sentence objects"""

    sentences = []  # contains Sentence objects
    s_idx = 0

    for line in file:
        if line[0].isdigit():                          # skip comments and blank lines
            l_split = line.split('\t')
            w_idx, word, pos = l_split[0], l_split[2], l_split[3]

            if w_idx == '1':                           # beginning of a new sentence
                s = Sentence(s_idx, [])
                sentences.append(s)
                s_idx += 1

            if '-' not in w_idx:                       # skip multi-word tokens
                sentences[-1].add_word(Word(w_idx, word, pos))

    return sentences


def build_dictionary(sentences):
    """
    Takes in a list of Sentence objects.
    Returns a dictionary where each word is assigned a unique integer key
    """
    words_dic = {}
    key = 0
    for sentence in sentences:
        for w in sentence.words:
            if words_dic.get(w.word) is None:
                words_dic[w.word] = key
                key += 1

    return words_dic


def build_word_list(sentences):
    """
    Takes in a list of Sentence objects.
    Returns a list where each word is assigned a unique integer key
    """
    words_dic = {}
    key = 0
    for sentence in sentences:
        for w in sentence.words:
            if words_dic.get(w.word) is None:
                words_dic[w.word] = key
                key += 1

    return words_dic


def sentence_to_word_index(words_dic, sentence):
    """
    Takes in a dictionary (built with the build_dictionary function) and
    a Sentence object.
    Returns the 1d-tensor of keys associated to the words of 'sentence'
    IF THE WORD IS UNKOWN, it adds it to the dictionary (with a new index)
    """
    for w in sentence.words:
        if w.word not in words_dic:
            words_dic[w.word] = len(words_dic)
            if glob.verbose:
                print("new word encountered :", w.word)

    return torch.tensor([words_dic[w.word] for w in sentence.words], device=glob.device)