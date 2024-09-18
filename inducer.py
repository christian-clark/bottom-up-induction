import torch
from copy import deepcopy
from itertools import permutations as perm
from torch import nn

from component_models import (
    CoocOpModel,
    cooc_op_five_word,
    fixed_op_three_word,
    hacky_operation_model,
    hacky_ordering_model
)
from tree import enumerate_trees

MAX_ROLE = 2

class Inducer(nn.Module):
    def __init__(self, config, learn_vectors, fixed_vectors, cooccurrences=None):
        super(Inducer, self).__init__()
        self.learn_vectors = learn_vectors
        self.fixed_vectors = fixed_vectors
        self.cooccurrences = cooccurrences
        self.vocab_size = fixed_vectors.shape[0]
        self.dvec = fixed_vectors.shape[1]
        self.emb = torch.nn.Embedding(self.vocab_size, self.dvec)
        print(self.emb)
        # operation probabilities
        # - input: functor and argument vectors (dim: 2*d)
        # - output: probability of composition operations:
        #       arg1, arg2, modification, or noop
        # TODO convert string directly into function?
        self.operation_model_type = config["operation_model_type"]
        if self.operation_model_type == "mlp":
            self.operation_model = nn.Linear(2*self.dvec, 4)
        elif self.operation_model_type == "cooc_op":
            self.operation_model = CoocOpModel(self.dvec, cooccurrences)
            #self.operation_model = cooc_op_five_word
        elif self.operation_model_type == "fixed_op_three_word":
            self.operation_model = fixed_op_three_word
        elif self.operation_model_type == "hacky":
            self.operation_model = hacky_operation_model
        else:
            raise NotImplementedError()
        # ordering probabilities
        # - input: one-hot composition operation (dim: omega, hardcoded as 3)
        #   - operations:
        #       - 0: arg1 attachment
        #       - 1: arg2 attachment
        #       - 2: modifier attachment
        # - output: probability of functor on left given composition operation
        self.ordering_model_type = config["ordering_model_type"]
        if self.ordering_model_type == "mlp":
            self.ordering_model = nn.Linear(3, 1)
        else:
            self.ordering_model = hacky_ordering_model

    def vectorize_sentence(self, ids):
        learned_vec = torch.softmax(self.emb(ids), dim=1)
        learn = self.learn_vectors.gather(dim=0, index=ids).unsqueeze(dim=1)
        repeated_ids = ids.unsqueeze(dim=1).repeat(1, self.dvec)
        fixed_vec = self.fixed_vectors.gather(dim=0, index=repeated_ids)
        combined_vec = learned_vec*learn + fixed_vec
        return combined_vec

    def forward(self, x, print_trees=False, verbose=True, get_tree_strings=False):
        """x is an n x d vector containing the d-dimensional vectors for a
        sentence of length n"""
        x = self.vectorize_sentence(x)
        if len(x) == 1:
            raise Exception("single-word sentences not supported")
        nonterminals = list()
        valid = list()
        ix = 0
        tree_strings = list()
        for t in enumerate_trees(x):
            if print_trees:
                print("================ TREE {} ================".format(ix))
                print(t.root)
                print(t)
                print()
            if get_tree_strings:
                tree_strings.append(str(t.root) + "\n" + str(t))
            nonterminals.append(t.nonterminals)
            if not t.root.separable or t.max_func_chain > MAX_ROLE:
                valid.append(0)
            else:
                valid.append(1)
            ix += 1

        func_vecs = list()
        arg_vecs = list()
        operations = list()
        operations_one_hot = list()
        directions = list()

        #for t in nonterminals[:10]:
        for t in nonterminals:
            curr_func_vecs = list()
            curr_arg_vecs = list()
            curr_ops = list()

            curr_dirs = list()
            for nt in t:
                curr_func_vecs.append(nt.func.vector)
                curr_arg_vecs.append(nt.arg.vector)

                if nt.op == "A":
                    if nt.func_chain == 1:
                        # arg1
                        curr_ops.append(0)
                    # NOTE: this also catches longer functor chains. But these
                    # will be caught as not valid
                    else:
                        # arg2
                        curr_ops.append(1)
                elif nt.op == "M":
                    curr_ops.append(2)
                # for the reverse direction - noop
                if nt.functor_position == "L":
                    curr_dirs.append(0)
                # NOTE: this also catches cases where functor position
                # is X, i.e. for permutations that aren't separable.
                # But these will be caught as not valid
                else:
                    curr_dirs.append(1)
            curr_func_vecs = torch.stack(curr_func_vecs, dim=0)
            func_vecs.append(curr_func_vecs)
            curr_arg_vecs = torch.stack(curr_arg_vecs, dim=0)
            arg_vecs.append(curr_arg_vecs)
            curr_ops = torch.tensor(curr_ops)
            # 3 classes for arg1, arg2, mod
            curr_ops_one_hot = torch.nn.functional.one_hot(
                curr_ops, num_classes=3
            ).float()
            operations.append(curr_ops)
            operations_one_hot.append(curr_ops_one_hot)
            curr_dirs = torch.tensor(curr_dirs)
            directions.append(curr_dirs)

        func_vecs = torch.stack(func_vecs, dim=0)
        arg_vecs = torch.stack(arg_vecs, dim=0)
        operations = torch.stack(operations, dim=0).unsqueeze(dim=-1)
        operations_one_hot = torch.stack(operations_one_hot, dim=0)
        directions = torch.stack(directions, dim=0).unsqueeze(dim=-1)
        op_input = torch.cat([func_vecs, arg_vecs], dim=2)
        op_scores = self.operation_model(op_input)
#        if self.operation_model_type == "mlp":
#            op_probs = torch.softmax(op_probs, dim=2)
        observed_op_scores = op_scores.gather(dim=2, index=operations).squeeze(dim=-1)
        nt_count = observed_op_scores.shape[1]
        pred_tree_scores = torch.exp(observed_op_scores.sum(dim=1)/nt_count)
        valid = torch.Tensor(valid)

        order_probs = self.ordering_model(operations_one_hot)
        if self.ordering_model_type == "mlp":
            order_probs = torch.sigmoid(order_probs)
        
        # this has the effect of replacing Pr(L | operation) with
        # Pr(R | direction) at nodes where the functor is on the right
        observed_order_probs = order_probs + (directions * (-2*order_probs + 1))
        observed_order_probs = observed_order_probs.squeeze(dim=-1)
        word_order_probs = observed_order_probs.prod(dim=1) * valid

        combined_scores = pred_tree_scores * word_order_probs
        if len(combined_scores) > 10:
            top = torch.topk(combined_scores, 10, dim=0)
        else:
            top = torch.sort(combined_scores, descending=True)
        loss = -1 * torch.log(combined_scores.sum())
        return loss, top
    