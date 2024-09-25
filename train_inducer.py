import csv, sys, torch
from configparser import ConfigParser
from copy import deepcopy
from torch import nn
from itertools import permutations

from inducer import Inducer
from tree import full_binary_trees

DEBUG = False
DEFAULT_CONFIG = {
    "DEFAULT": {
        "max_epoch": 1000,
        "epoch_print_freq": 100,
        "operation_model_type": "mlp",
        "ordering_model_type": "mlp"
    }
}

def printDebug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


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
    word2ix = dict()
    # read corpus to get vocab
    corpus = list()
    for sent in open(config["training_data"]):
        sent_ids = list()
        for w in sent.strip().split():
            if w not in word2ix:
                word2ix[w] = len(word2ix)
            sent_ids.append(word2ix[w])
        corpus.append(torch.tensor(sent_ids))

    print("word2ix:", word2ix)
    vocab_size = len(word2ix)
    vdim = int(config["vector_dim"])
    fixed_vectors = torch.zeros(vocab_size, vdim)
    learn_vectors = torch.ones(vocab_size)
    if "fixed_vectors" in config and config["fixed_vectors"] != "None":
        for row in csv.reader(open(config["fixed_vectors"])):
            w = row[0]
            # ignore lines starting with #
            # ignore fixed vectors for words if they aren't in the corpus
            if w.startswith("#") or w not in word2ix:
                continue
            ix = word2ix[w]
            learn_vectors[ix] = 0
            v = row[1:]
            assert len(v) == vdim
            v = torch.Tensor([float(x) for x in v])
            fixed_vectors[ix] = v

    return corpus, fixed_vectors, learn_vectors


def get_cooccurrences(config):
    if "cooccurrence_scores_dir" not in config:
        return None
    cooc = list()
    dir = config["cooccurrence_scores_dir"]
    #for op in ["argument1", "argument2", "modifier", "null"]:
    for op in ["argument1", "argument2", "modifier"]:
        op_scores = list()
        for row in csv.reader(open(dir + '/' + op)):
            scores = [float(s) for s in row]
            op_scores.append(scores)
        cooc.append(op_scores)
    # shape after permuute: pred x pred x op
    cooc = torch.Tensor(cooc).permute((1, 2, 0))
    #cooc = torch.softmax(cooc, dim=2)
    return cooc


def tree_strings_from_backpointers(backpointers):
    strings = list()
    sent_len = backpointers.shape[0]
    for bp_info in backpointers[sent_len-1,0]:
        s = _construct_tree_string(backpointers, bp_info, 0, sent_len-1)
        strings.append(s)
    return strings


def _construct_tree_string(backpointers, bp_info, start, end):
    ijdiff = bp_info[0].long().item() 
    left_beam_ix = bp_info[1].long().item()
    right_beam_ix = bp_info[2].long().item()
    dir = bp_info[3].item()
    op = bp_info[4].item()
    # backpointer items are set as -1 for leaves
    if ijdiff == -1:
        return str(start) 
    else:
        # left child spans start to start+split, inclusive
        left_start = start
        left_delta = ijdiff
        left_end = start + ijdiff
        printDebug("left_start type:", type(left_start))
        printDebug("left_delta type:", type(left_delta))
        printDebug("left_end type:", type(left_end))
        printDebug("left_beam_ix type:", type(left_beam_ix))
        left_bp_info = backpointers[left_delta,left_start,left_beam_ix]
        left_string = _construct_tree_string(
            backpointers, left_bp_info, left_start, left_end
        )

        right_start = start + ijdiff + 1
        right_delta = end - start - ijdiff - 1
        right_end = end
        right_bp_info = backpointers[right_delta,right_start,right_beam_ix]
        right_string = _construct_tree_string(
            backpointers, right_bp_info, right_start, right_end
        )

        # functor is on left
        if op == 0:
            ostr = "A1"
        elif op == 1:
            ostr = "A2"
        else:
            assert op == 2
            ostr = "M"
        if dir == 0:
            return '(' + ' ' + left_string + ' ' + right_string + ' ' + ')' + ostr
        # functor is on right
        else:
            assert dir == 1
            return '(' + ' ' + right_string + ' ' + left_string + ' ' + ')' + ostr


def train_inducer(config):
    # dim: sents x words x d
    # d is the dimension of the word vectors
    corpus, fixed_vectors, learn_vectors = get_corpus_and_embeddings(config)
    cooccurrences = get_cooccurrences(config)
    inducer = Inducer(config, learn_vectors, fixed_vectors, cooccurrences)
    optimizer = torch.optim.Adam(inducer.parameters())
    epoch = 0
    loss_tracking = list()
    i2t = IxToTree()
    torch.set_printoptions(linewidth=200, precision=2)
    while epoch < config.getint("max_epoch"):
        optimizer.zero_grad()
        per_sent_loss = list()
        # TODO batching
        if epoch % config.getint("epoch_print_freq") == 0:
            print("\n==== Epoch {} ====".format(epoch))
            print_results = True
        else:
            print_results = False

        for sent_ix, sent in enumerate(corpus):
            if print_results:
                sent_loss, tree_scores, backpointers = inducer(sent, return_backpointers=True)
                # TODO print trees based on backpointers
                trees = tree_strings_from_backpointers(backpointers)
                print("\n\t== Sentence {} ==".format(sent_ix))
                for ix, t in enumerate(trees):
                    score = tree_scores[ix].item()
                    print("\t" + t + "\tScore: {}".format(round(score, 4)))
            else:
                sent_loss, _ = inducer(sent)
            per_sent_loss.append(sent_loss)
        loss = sum(per_sent_loss)
        if print_results:
            learned_emb_softmax = torch.softmax(inducer.emb.weight, dim=1)
            combined_emb = \
                learned_emb_softmax*inducer.learn_vectors.unsqueeze(dim=1) \
                    + inducer.fixed_vectors
            print("\nembeddings:")
            print(combined_emb)
            ord_model = inducer.ordering_model
            ops = nn.functional.one_hot(torch.arange(3)).float()
            ord_probs = torch.sigmoid(ord_model(ops))
            print("\nordering probs:")
            print(ord_probs.reshape(-1))
            print("\nloss:", loss)
            loss_tracking.append(loss)
        loss.backward()
        optimizer.step()
        epoch += 1
    
    loss_tracking = torch.stack(loss_tracking, dim=0)
    print("\n==== Loss ====")
    print(loss_tracking)


if __name__ == "__main__":
    if sys.argv[1] == "-h":
        print("Usage: train_inducer.py [config] [overrides]")
    else:
        config = get_config()
        print(dict(config))
        train_inducer(config)
