import csv, sys, torch
from configparser import ConfigParser
from copy import deepcopy
from itertools import permutations

from inducer import Inducer
from tree import full_binary_trees


DEFAULT_CONFIG = {
    "DEFAULT": {
        "max_epoch": 1000,
        "epoch_print_freq": 100,
        "operation_model_type": "mlp",
        "ordering_model_type": "mlp"
    }
}


class IxToTree:
    def __init__(self):
        self.tree_structures = dict()
        self.permutations = dict()

    def get_tree(self, ix, sent_len):
        if sent_len not in self.tree_structures:
            assert sent_len not in self.permutations
            self._store_tree_structures(sent_len)
            self._store_permutations(sent_len)
        struct_ix = ix // len(self.permutations[sent_len])
        perm_ix = ix % len(self.permutations[sent_len])
        struct = self.tree_structures[sent_len][struct_ix]
        perm = self.permutations[sent_len][perm_ix]
        tree = deepcopy(struct)
        tree.set_leaf_nodes(list(perm))
        return tree

    def _store_tree_structures(self, sent_len):
        self.tree_structures[sent_len] = full_binary_trees(2*sent_len-1)

    def _store_permutations(self, sent_len):
        self.permutations[sent_len] = list(permutations(range(sent_len)))


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


def get_corpus_and_embeddings(config):
    #word_vectors = dict()
    # config["word_vectors"] is a CSV
    word2ix = dict()
    # read corpus to get vocab
    #vectorized_corpus = list()
    corpus = list()
    for sent in open(config["training_data"]):
        sent_ids = list()
        #v_sent = list()
        for w in sent.strip().split():
            if w not in word2ix:
                word2ix[w] = len(word2ix)
            sent_ids.append(word2ix[w])
        corpus.append(torch.tensor(sent_ids))

    print("word2ix:", word2ix)

    # read fixed vectors
    vocab_size = len(word2ix)
    vdim = int(config["vector_dim"])
    fixed_vectors = torch.zeros(vocab_size, vdim)
    learn_vectors = torch.ones(vocab_size)
    if "fixed_vectors" in config:
        for row in csv.reader(open(config["fixed_vectors"])):
            w = row[0]
            # ignore fixed vectors for words if they aren't in the corpus
            if w not in word2ix:
                continue
            ix = word2ix[w]
            learn_vectors[ix] = 0
            v = row[1:]
            assert len(v) == vdim
            v = torch.Tensor([float(x) for x in v])
            fixed_vectors[ix] = v

    #emb = torch.nn.Embedding(len(word2ix), vdim)

    return corpus, fixed_vectors, learn_vectors


def get_cooccurrences(config):
    if "cooccurrence_scores_dir" not in config:
        return None
    cooc = list()
    dir = config["cooccurrence_scores_dir"]
    for op in ["argument1", "argument2", "modifier", "null"]:
        op_scores = list()
        for row in csv.reader(open(dir + '/' + op)):
            scores = [float(s) for s in row]
            op_scores.append(scores)
        cooc.append(op_scores)
    cooc = torch.Tensor(cooc).permute((1, 2, 0))
    cooc = torch.softmax(cooc, dim=2)
    return cooc


if __name__ == "__main__":
    config = get_config()
    print(dict(config))
    # dim: sents x words x d
    # d is the dimension of the word vectors
    corpus, fixed_vectors, learn_vectors = get_corpus_and_embeddings(config)
    #corpus, dim_v = get_training_data(config)
    print(corpus)
    print(fixed_vectors)
    print(learn_vectors)
    cooccurrences = get_cooccurrences(config)
    inducer = Inducer(config, learn_vectors, fixed_vectors, cooccurrences)
    #inducer = Inducer(x.shape[1], ordering_model_type="hacky")
    #inducer = Inducer(x.shape[1])
    # default lr: 0.001
    optimizer = torch.optim.Adam(inducer.parameters())
    epoch = 0
    #loss, combined_probs = inducer(x, print_trees=True)
    #loss, combined_probs, tree_strings = inducer(x, get_tree_strings = True)
    #loss = inducer(corpus, get_tree_strings = True)
#    for ix, s in enumerate(tree_strings):
#        print("\n======== TREE {} ========".format(ix))
#        print(s)
#    for ix in [448, 168, 552, 360, 266, 936, 744, 462]:
#        print("\n======== TREE {} ========".format(ix))
#        print(tree_strings[ix])
    #combined_probs_tracking = list()
    loss_tracking = list()
    i2t = IxToTree()
    torch.set_printoptions(linewidth=200, precision=2)
    while epoch < config.getint("max_epoch"):
        optimizer.zero_grad()
        if epoch % config.getint("epoch_print_freq") == 0: v = True
        else: v = False
        #loss, combined_probs = inducer(x, verbose=v)
        per_sent_loss = list()
        per_sent_top = list()
        for sent in corpus:
            sent_loss, top = inducer(sent, verbose=v)
            per_sent_loss.append(sent_loss)
            per_sent_top.append(top)
        loss = sum(per_sent_loss)
        loss.backward()
        optimizer.step()
        if epoch % config.getint("epoch_print_freq") == 0:
            print("\n==== Epoch {} ====".format(epoch))
            print("loss:", loss)
            #print("embeddings:", torch.softmax(inducer.emb.weight, dim=1))
            learned_emb_softmax = torch.softmax(inducer.emb.weight, dim=1)
            combined_emb = \
                learned_emb_softmax*inducer.learn_vectors.unsqueeze(dim=1) \
                    + inducer.fixed_vectors
            print("embeddings:")
            print(combined_emb)
            #print("tree probabilities:", combined_probs)
            #print_tree_probabilities(combined_probs, sort=True, top_k=15)
            #combined_probs_tracking.append(combined_probs)
            for i, top in enumerate(per_sent_top):
                print("Top trees for sentence {}:".format(i+1))
                indices = top.indices
                values = top.values
                for j, ix in enumerate(indices[:5]):
                    print("\t[{}] {}".format(j+1, i2t.get_tree(ix, len(corpus[i]))))
                    print("\t\tProb =", round(values[j].item(), 4))
            loss_tracking.append(loss)
        epoch += 1
    
    #combined_probs_tracking = torch.stack(combined_probs_tracking, dim=0)
    #print("\n\n==== Tree probabilities ====")
    #print(combined_probs_tracking)
    
    loss_tracking = torch.stack(loss_tracking, dim=0)
    print("\n==== Loss ====")
    print(loss_tracking)