import csv, sys, torch
from configparser import ConfigParser
from inducer import Inducer

UNK = "<unk>"

DEFAULT_CONFIG = {
    "DEFAULT": {
        "max_epoch": 1000,
        "epoch_print_freq": 100,
        "operation_model_type": "mlp",
        "ordering_model_type": "mlp"
    }
}

def print_tree_probabilities(probs, sort=False, top_k=None):
    probs = list((ix, p.item()) for ix, p in enumerate(probs))
    print("Tree probabilities:")
    if sort:
        probs = sorted(probs, key=lambda x:x[1], reverse=True)
    if top_k:
        probs = probs[:top_k]
    for ix, p in probs:
        print("\t{}\t{:0.3e}".format(ix, p))


# word vectors:
#
#           animate eating  edible
# people    1       0       0
# eat       0       1       0
# donuts    0       0       1

#role1(func, arg):
#func[eating] AND arg[animate]

#role2(func, arg):
#func[eating] AND arg[edible]

#composition: just keep functor vector


# people eat donuts
# or
# happy people eat
#x = torch.Tensor([
#    [1, 0, 0],
#    [0, 1, 0],
#    [0, 0, 1]
#])

#x = torch.Tensor([
#    [0.1, 0.02, -0.03],
#    [-0.02, 0.2, 0.01],
#    [-0.01, 0.03, 0.3]
#])

# words:
# 0: happy
# 1: people
# 2: eat
# 3: yummy
# 4: donuts

# happy people eat donuts
#x = torch.Tensor([
#    [1, 0, 0, 0, 0],
#    [0, 1, 0, 0, 0],
#    [0, 0, 1, 0, 0],
#    [0, 0, 0, 0, 1]
#])
#

def get_config():
    top_config = ConfigParser()
    top_config.read_dict(DEFAULT_CONFIG)
    # no config file or overrides
    if len(sys.argv) == 1:
        overrides = []
    # config file, and possibly overrides
    elif len(sys.argv[1].split("=")) == 1:
        top_config.read(sys.argv[1])
        overrides = sys.argv[2:]
    # just overrides
    else:
        overrides = sys.argv[1:]
    config = top_config["DEFAULT"]

    # any args after the config file override key-value pairs
    for kv in overrides:
        k, v = kv.split("=")
        config[k] = v
    return config


def get_training_data(config):
    word_vectors = dict()
    # config["word_vectors"] is a CSV
    for row in csv.reader(open(config["word_vectors"])):
        w = row[0]
        v = row[1:]
        v = torch.Tensor([float(x) for x in v])
        word_vectors[w] = v

    vectorized_corpus = list()
    for sent in open(config["training_data"]):
        v_sent = list()
        for w in sent.strip().split():
            if w in word_vectors:
                v_sent.append(word_vectors[w])
            else:
                v_sent.append(word_vectors[UNK])
        vectorized_corpus.append(torch.stack(v_sent))
    return torch.stack(vectorized_corpus)


if __name__ == "__main__":
    config = get_config()
    #print(dict(config))
    x = get_training_data(config)
    print(x)
    inducer = Inducer(x.shape[1], config)
    #inducer = Inducer(x.shape[1], ordering_model_type="hacky")
    #inducer = Inducer(x.shape[1])
    # default lr: 0.001
    optimizer = torch.optim.Adam(inducer.parameters())
    epoch = 0
    #loss, combined_probs = inducer(x, print_trees=True)
    loss, combined_probs, tree_strings = inducer(x, get_tree_strings = True)
    for ix, s in enumerate(tree_strings):
        print("\n======== TREE {} ========".format(ix))
        print(s)
#    for ix in [448, 168, 552, 360, 266, 936, 744, 462]:
#        print("\n======== TREE {} ========".format(ix))
#        print(tree_strings[ix])
    combined_probs_tracking = list()
    loss_tracking = list()
    torch.set_printoptions(linewidth=200, precision=2)
    while epoch < config.getint("max_epoch"):
        if epoch % config.getint("epoch_print_freq") == 0:
            print("\n==== Epoch {} ====".format(epoch))
            print("loss:", loss)
            #print("tree probabilities:", combined_probs)
            print_tree_probabilities(combined_probs, sort=True, top_k=15)
            combined_probs_tracking.append(combined_probs)
            loss_tracking.append(loss)
        optimizer.zero_grad()
        if epoch % config.getint("epoch_print_freq") == 0: v = True
        else: v = False
        loss, combined_probs = inducer(x, verbose=v)
        loss.backward()
        optimizer.step()
        epoch += 1
    
    combined_probs_tracking = torch.stack(combined_probs_tracking, dim=0)
    #print("\n\n==== Tree probabilities ====")
    #print(combined_probs_tracking)
    
    loss_tracking = torch.stack(loss_tracking, dim=0)
    print("\n==== Loss ====")
    print(loss_tracking)