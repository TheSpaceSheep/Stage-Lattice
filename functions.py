import torch
import torch.nn as nn
import global_variables as glob
import os


def train(net, batch, actual_tags, optimizer):
    net.train()
    optimizer.zero_grad()
    output = net.forward(batch)
    loss = nn.CrossEntropyLoss()
    out = loss(output, actual_tags)
    out.backward()
    optimizer.step()


def save_network(Net, save_path):
    """saves the Model Net,
    specifying on which dataset it has been trained (name of the file)
    and for how many epochs.
    """
    path = save_path.split('/')     # path = ["dir", "subdir", "file_name"]
    path.pop(-1)                    # path = ["dir", "subdir"]
    s = ""
    for d in path:
        s = s + (s != "") * '/' + d
        if not os.path.isdir(s):    # create directories if they don't exist
            os.mkdir(s)

    save_file = open(glob.SAVE_NET_PATH, 'w+')
    torch.save(Net.state_dict(), glob.SAVE_NET_PATH)
    save_file.close()
