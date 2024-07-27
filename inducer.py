import torch
from copy import deepcopy
from itertools import permutations as perm
from torch import nn

from tree import Tree, full_binary_trees


MAX_ROLE = 2

def enumerate_trees(x):
    # generate all tree structures (catalan)
    # loop through and assign all permutations (factorial)
    # yield one at a time
    sent_len = x.shape[0]
    tree_structures = full_binary_trees(2*sent_len-1)
    for ts in tree_structures:
        ps = perm(range(sent_len))
        for p in ps:
            ts_curr = deepcopy(ts)
            ts_curr.set_leaf_nodes(list(p))
            t = Tree(ts_curr, vectors=list(x))
            yield t
    

class Inducer(nn.Module):
    def __init__(self, vector_dim):
        super(Inducer, self).__init__()
        self.d = vector_dim
        # components:
        # - operation probabilities
        #   - input: functor and argument vectors (dim: 2*d)
        #   - output: probability for each possible composition operation
        #       (each value between 0 and 1)
        self.operation_mlp = nn.Linear(2*self.d, 2)
        # - ordering probabilities
        #   - input: one-hot composition operation (dim: omega)
        #       and direction (left or right, dim: 1)              
        #   - output: probability of direction given operation
        # TODO This doesn't work because it doesn't guarantee that pr(left) = 1 - pr(right)
        # should be:
        # - input: one-hot composition operation (dim: omega)
        # - output: probability of functor on left given composition operation
        # then you either take raw output or 1 - output depending on the nonterminal
        # NOTE: 2 composition operations is currently hardcoded
        self.ordering_mlp = nn.Linear(2, 1)

    def forward(self, x):
        """x is an n x d vector containing the d-dimensional vectors for a
        sentence of length n"""
        nonterminals = list()
        valid = list()
        ix = 0
        for t in enumerate_trees(x):
            print("================ TREE {} ================".format(ix))
            print(t.root)
            print(t)
            print("max func chain:", t.max_func_chain)
            nonterminals.append(t.nonterminals)
            #print("score:", t.unnormalized_score())
            if not t.root.separable or t.max_func_chain > MAX_ROLE:
                valid.append(0)
            else:
                valid.append(1)
            ix += 1
            print()

        func_vecs = list()
        arg_vecs = list()
        operations = list()
        operations_one_hot = list()
        directions = list()
        for t in nonterminals:
            curr_func_vecs = list()
            curr_arg_vecs = list()
            curr_ops = list()
            curr_dirs = list()
            for nt in t:
                curr_func_vecs.append(nt.func.vector)
                curr_arg_vecs.append(nt.arg.vector)
                # one-hot encoding of composition operation
                if nt.func_chain == 1:
                    #curr_ops.append(torch.Tensor([1, 0]))
                    curr_ops.append(0)
                # NOTE: this also catches longer functor chains. But these
                # will be caught as not valid
                else:
                    #curr_ops.append(torch.Tensor([0, 1]))
                    curr_ops.append(1)
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
            #curr_ops = torch.stack(curr_ops, dim=0)
            curr_ops = torch.tensor(curr_ops)
            curr_ops_one_hot = torch.nn.functional.one_hot(
                curr_ops, num_classes=MAX_ROLE
            ).float()
            operations.append(curr_ops)
            operations_one_hot.append(curr_ops_one_hot)
            curr_dirs = torch.tensor(curr_dirs)
            directions.append(curr_dirs)

        func_vecs = torch.stack(func_vecs, dim=0)
        arg_vecs = torch.stack(arg_vecs, dim=0)
        valid = torch.Tensor(valid)
        operations = torch.stack(operations, dim=0).unsqueeze(dim=-1)
        operations_one_hot = torch.stack(operations_one_hot, dim=0)
        directions = torch.stack(directions, dim=0).unsqueeze(dim=-1)

        op_input = torch.cat([func_vecs, arg_vecs], dim=2)
        op_scores = self.operation_mlp(op_input)
        op_probs = torch.sigmoid(op_scores)
        observed_op_probs = op_probs.gather(dim=2, index=operations).squeeze(dim=-1)
        pred_tree_scores = observed_op_probs.prod(dim=1)
        pred_tree_total = pred_tree_scores.sum().item()
        pred_tree_probs = pred_tree_scores / pred_tree_total

        #order_input = torch.cat([operations_one_hot, directions], dim=2)
        order_scores = self.ordering_mlp(operations_one_hot)
        order_probs = torch.sigmoid(order_scores)
        # this has the effect of replacing Pr(L | operation) with
        # Pr(R | direction) at nodes where the functor is on the right
        observed_order_probs = order_probs + (directions * (-2*order_probs + 1))
        observed_order_probs = observed_order_probs.squeeze(dim=-1)
        word_order_probs = observed_order_probs.prod(dim=1) * valid

        combined_probs = pred_tree_probs * word_order_probs
        print("combined_probs:")
        print(combined_probs)



        # TODO
        # - gather from op_probs according to the operation that occurs at
        # each nonterminal
        # - multiply across nonterminals to get scores for predicate trees
        # and word orders
        # - normalize by the sum of predicate tree scores
        # - multiply in valid to filter out bad sentences


