from functions import train, save_network
from preprocessing import read_file, build_dictionary, sentence_to_word_index
from models import WordEmb, LinearPosTagger, LSTMPosTagger, BiLSTMPosTagger, \
    BiLSTMPosTaggerEXPERIMENT, MultiLayerBiLSTMPosTagger, MultiLayerBiLSTMPosTaggerHOMEMADE, \
    XLMEmbMultiLayerBiLSTMPosTagger
import XLM_pre
import global_variables as glob
import torch
import time
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "force_cpu":
        glob.device = "cpu"

print("using device :", glob.device)

""" ___________________________________ PRE-PROCESSING __________________________________________ """

if glob.multilingual:
    files = [open("datasets/en_ewt-ud-{}.conllu".format(glob.train_mode), 'r'),
             open("datasets/fr_gsd-ud-{}.conllu".format(glob.train_mode), 'r'),
             open("datasets/es_gsd-ud-{}.conllu".format(glob.train_mode), 'r')]

    files_test = [open("datasets/en_ewt-ud-{}.conllu".format(glob.test_mode), 'r'),
                  open("datasets/fr_gsd-ud-{}.conllu".format(glob.test_mode), 'r'),
                  open("datasets/es_gsd-ud-{}.conllu".format(glob.test_mode), 'r')]

    sentences = []
    sentences_test = []
    for f, f_test in zip(files, files_test):
        sentences += read_file(f)
        sentences_test += read_file(f_test)

else:
    raw_file = open("datasets/fr_gsd-ud-{}.conllu".format(glob.train_mode), 'r')
    raw_file_test = open("datasets/fr_gsd-ud-{}.conllu".format(glob.test_mode), 'r')

    sentences = read_file(raw_file)
    sentences_test = read_file(raw_file_test)

dico = build_dictionary(sentences)

"""___________________________________ INITIALIZING MODEL _______________________________________"""

Net = XLMEmbMultiLayerBiLSTMPosTagger(dico)
Net.to(glob.device)

if glob.LOAD_NET_PATH != "":
    try:
        Net.load_state_dict(torch.load(glob.LOAD_NET_PATH))
    except FileNotFoundError:
        print("ERROR : Couldn't load weights (File Not Found)")

optimizer = torch.optim.SGD(Net.parameters(), lr=0.01, momentum=0.9)

"""_______________________________________ TRAINING _____________________________________________"""

# for epochs in [1, 2, 3, 4]:
# for epochs in [10]:
if glob.epochs:
    print("epoch : ", end="")
    # glob.epochs = epochs
    learning_time = time.clock()
    try:
        for epoch in range(glob.epochs):
            print(epoch + 1)
            print("progress : ")

            sys.stdout.flush()
            nb_batches = len(sentences)//glob.batch_size
            start_time = time.clock()

            for i_batch in range(nb_batches - 1):

                sentences_batch = sentences[i_batch * glob.batch_size:(i_batch+1) * glob.batch_size]

                # xlm_batch = XLM_pre.create_batch(sentences_batch)
                # xlm_batch = xlm_batch.detach()       # do not train the pretrained XLM weights

                for s in sentences_batch:
                    actual_tags = torch.tensor([glob.POS_DICT[w.pos] for w in s.words], device=glob.device)
                    # indexes = sentence_to_word_index(dico, s)
                    train(Net, s, actual_tags, optimizer)

                batch_time = time.clock() - start_time
                print(round(100*(i_batch + 1)/nb_batches, 2), end="% ")
                remaining_time = batch_time/((i_batch + 1)/nb_batches) - batch_time
                remaining_time *= glob.epochs
                h = int(remaining_time // 3600)
                m = int(remaining_time // 60 - (remaining_time // 3600) * 60)
                s = int(remaining_time % 60)
                print("Remaining : {0}h{1}m{2}s".format(h, m, s))
                sys.stdout.flush()

            print('.')

    except KeyboardInterrupt:
        print('\n' + '*' * 50)
        print(' ' * 10 + "Exiting from training early")
        print('*' * 50)
    learning_time = time.clock() - learning_time

# ------------------------------- Saving parameters after training -------------------------------

if glob.SAVE_NET_PATH != "":
    save_network(Net, glob.SAVE_NET_PATH)


"""________________________________________ TESTING ______________________________________________"""

with torch.no_grad():
    accuracy = []
    for s in sentences_test:
        # indexes = sentence_to_word_index(dico, s)
        values, predicted_pos = Net.predict(s)
        err = [s.words[i].pos == predicted_pos[i] for i in range(len(predicted_pos))]
        accuracy += [sum(err) / len(s.words)]

        if glob.verbose:
            print(s)
            print(predicted_pos)
            print(sum(err), "/", len(s.words))


print("Accuracy : ", torch.tensor(accuracy).mean().item())

if glob.epochs:
    h = int(learning_time // 3600)
    m = int(learning_time // 60 - (learning_time // 3600) * 60)
    s = int(learning_time % 60)
    print("Learning time : {0}h{1}m{2}s".format(h, m, s))


"""______________________________________ CLOSING FILES ___________________________________________"""

if glob.multilingual:
    for file in files:
        file.close()
    for file_test in files_test:
        file_test.close()
else:
    raw_file.close()
    raw_file_test.close()
