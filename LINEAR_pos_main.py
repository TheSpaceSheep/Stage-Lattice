import torch
import global_variables as glob
from models import LinearPosTagger, LSTMPosTagger, MultiLayerBiLSTMPosTagger
from functions import train
import time
import sys

"""
#### Training times ('train' dataset) ####

-------------------------------------------------
             |    CUDA        |       CPU
-------------------------------------------------                
1 epoch      |    1'30"       |       1'38"
2 epochs     |    2'19"       |       5'47"
3 epochs     |    3'21"       |       8'10
4 epochs     |    4'26"       |     not tested
10 epochs    |    11'09"      |     not tested
-------------------------------------------------


#### Accuracies ####

using the 'train' dataset :

-> EMB_DIM = 100

No training    2-6 %
--------------------
1 epoch        ~57 %
2 epochs       ~60 %
3 epochs       ~58 %
4 epochs       ~57 %
10 epochs      ~57 %


-> EMB_DIM = 50

No training    2-11 %
---------------------
1 epoch        64-66 %
2 epochs       ~66 %


using the 'dev' dataset (EMB_DIM = 100) :

No training    2-10 %
1 epoch        55-60 %
2 epochs       61-67 %
3 epochs       67-69 %
4 epochs       65-69 %
10 epochs      69-71 %
"""

if sys.argv[0] == "force_cpu":
    glob.device = "cpu"

Net = LinearPosTagger()
Net.to(glob.device)

optimizer = torch.optim.SGD(Net.parameters(), lr=0.01, momentum=0.9)


""" ___________________________________ PRE-PROCESSING __________________________________________ """

raw_file = open("datasets/fr_gsd-ud-dev.conllu", 'r')
raw_file_test = open("datasets/fr_gsd-ud-test.conllu", 'r')


sentences = read_file(raw_file)
sentences_test = read_file(raw_file_test)
dico = build_dictionary(sentences)

Embedding = WordEmb(len(dico), glob.EMB_DIM)
Embedding.to(glob.device)


"""_______________________________________ TRAINING _____________________________________________ """

learning_time = time.clock()

for epoch in range(glob.epochs):
    for s in sentences:
        indexes = sentence_to_word_index(dico, s)
        V = Embedding(indexes)
        actual_tags = torch.tensor([glob.POS_DICT[w.pos] for w in s.words], device=glob.device)

        train(Net, V, actual_tags, optimizer)

learning_time = time.clock() - learning_time


"""________________________________________ TESTING ______________________________________________"""

with torch.no_grad():
    accuracy = []
    for s in sentences_test:
        indexes = sentence_to_word_index(dico, s)
        V = Embedding(indexes)
        values, predicted_pos = Net.predict(V)
        err = [s.words[i].pos == predicted_pos[i] for i in range(len(predicted_pos))]
        print(s)
        print(err)
        print(sum(err), "/", len(s.words))
        accuracy += [sum(err) / len(s.words)]

print("Accuracy : ", torch.tensor(accuracy).mean().item())

if glob.epochs:
    h = int(learning_time//3600)
    m = int(learning_time//60 - (learning_time//3600)*60)
    s = int(learning_time % 60)
    print("Learning time : {0}h{1}m{2}s".format(h, m, s))
