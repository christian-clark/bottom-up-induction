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


def hacky_operation_model(func_and_arg, version="four_word"):
    """
    non-learnable operation mlp for proof-of-concept system
    in test_inducer.py
    possible operations: arg1 attachment, arg2 attachment, modification, none
    input dim: trees x nonterminals x (2*dvec)
    output dim: trees x nonterminals x 4
    """
    hacky_output = list()
    if version == "three_word":
        for tree in func_and_arg:
            t_output = list()
            for nont in tree:
                v_func = nont[:3].tolist()
                v_arg = nont[3:].tolist()
                arg1 = 0
                arg2 = 0
                modification = 0
                neither = 1
                # functor word is "eat"
                if int(v_func[1]) == 1:
                    # arg is "people"
                    if int(v_arg[0]):
                        arg1 = 1
                        neither = 0
                    # arg is "donuts"
                    elif int(v_arg[2]):
                        arg2 = 1
                        neither = 0
                t_output.append([arg1, arg2, modification, neither])
            hacky_output.append(t_output)
    elif version == "three_word_mod":
        for tree in func_and_arg:
            t_output = list()
            # happy people eat
            for nont in tree:
                v_func = nont[:3].tolist()
                v_arg = nont[3:].tolist()
                arg1 = 0
                arg2 = 0
                modification = 0
                neither = 1
                # functor word is "eat" and arg is "people"
                if int(v_func[2]) and int(v_arg[1]):
                    arg1 = 1
                    neither = 0
                # functor is "happy" and arg is "people"
                elif int(v_func[0]) and int(v_arg[1]):
                    # arg is "donuts"
                    modification = 1
                    neither = 0
                t_output.append([arg1, arg2, modification, neither])
            hacky_output.append(t_output)
    elif version == "four_word":
        for tree in func_and_arg:
            t_output = list()
            for nont in tree:
                v_func = nont[:5].tolist()
                v_arg = nont[5:].tolist()
                arg1 = 0
                arg2 = 0
                modification = 0
                neither = 1
                # functor word is "eat"
                if int(v_func[2]):
                    # arg is "people"
                    if int(v_arg[1]):
                        arg1 = 1
                        neither = 0
                    # arg is "donuts"
                    elif int(v_arg[4]):
                        arg2 = 1
                        neither = 0
                # functor is "happy" and arg is "people"
                elif int(v_func[0]) and int(v_arg[1]):
                    modification = 1
                    neither = 0
                # functor is "yummy" and arg is "donuts"
                elif int(v_func[3]) and int(v_arg[4]):
                    modification = 1
                    neither = 0
                t_output.append([arg1, arg2, modification, neither])
            hacky_output.append(t_output)
    else:
        raise NotImplementedError
    return torch.Tensor(hacky_output)


def hacky_ordering_model(operation):
    """
    non-learnable ordering model for proof-of-concept system in
    test_inducer.py
    input dim: trees x nonterminals x 3
    output dim: trees x nonterminals x 1
    """
    hacky_output = list()
    for tree in operation:
        t_output = list()
        for nont in tree:
            op = torch.argmax(nont).item()
            # op 0 is arg1 attachment; functor is on right
            if op == 0:
                t_output.append([0])
            # op 1 is arg2 attachment; functor is on left
            elif op == 1:
                t_output.append([1])
            # op 2 is modifier attachment; functor is on left
            else:
                assert op == 2, "unexpected op: {}".format(op)
                t_output.append([1])
        hacky_output.append(t_output)
    return torch.Tensor(hacky_output)
    

class Inducer(nn.Module):
    def __init__(self, vector_dim, operation_model_type="mlp", 
                 ordering_model_type="mlp"
        ):
        super(Inducer, self).__init__()
        self.d = vector_dim
        self.operation_model_type = operation_model_type
        self.ordering_model_type = ordering_model_type
        # operation probabilities
        # - input: functor and argument vectors (dim: 2*d)
        # - output: probability of composition operations:
        #       arg1, arg2, modification, or noop
        if self.operation_model_type == "mlp":
            self.operation_model = nn.Linear(2*self.d, 4)
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
        if self.ordering_model_type == "mlp":
            self.ordering_model = nn.Linear(3, 1)
        else:
            self.ordering_model = hacky_ordering_model

    def forward(self, x, print_trees=False, verbose=True, get_tree_strings=False):
        """x is an n x d vector containing the d-dimensional vectors for a
        sentence of length n"""
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

        good_func_vecs = list()
        good_arg_vecs = list()
        good_operations = list()
        good_operations_one_hot = list()

        bad_func_vecs = list()
        bad_arg_vecs = list()
        bad_operations = list()

        directions = list()

        # TODO loop through all trees
        #for t in nonterminals[:10]:
        for t in nonterminals:
            curr_good_func_vecs = list()
            curr_good_arg_vecs = list()
            curr_good_ops = list()

            curr_bad_func_vecs = list()
            curr_bad_arg_vecs = list()
            curr_bad_ops = list()

            curr_dirs = list()
            for nt in t:
                curr_good_func_vecs.append(nt.func.vector)
                curr_good_arg_vecs.append(nt.arg.vector)

                # for the reverse direction
                curr_bad_func_vecs.append(nt.arg.vector)
                curr_bad_arg_vecs.append(nt.func.vector)
                # one-hot encoding of composition operation
                if nt.op == "A":
                    if nt.func_chain == 1:
                        # arg1
                        curr_good_ops.append(0)
                    # NOTE: this also catches longer functor chains. But these
                    # will be caught as not valid
                    else:
                        # arg2
                        curr_good_ops.append(1)
                elif nt.op == "M":
                    curr_good_ops.append(2)
                # for the reverse direction - noop
                curr_bad_ops.append(3)
                if nt.functor_position == "L":
                    curr_dirs.append(0)
                # NOTE: this also catches cases where functor position
                # is X, i.e. for permutations that aren't separable.
                # But these will be caught as not valid
                else:
                    curr_dirs.append(1)
            curr_good_func_vecs = torch.stack(curr_good_func_vecs, dim=0)
            good_func_vecs.append(curr_good_func_vecs)
            curr_good_arg_vecs = torch.stack(curr_good_arg_vecs, dim=0)
            good_arg_vecs.append(curr_good_arg_vecs)
            curr_good_ops = torch.tensor(curr_good_ops)
            # 3 classes for arg1, arg2, mod
            curr_good_ops_one_hot = torch.nn.functional.one_hot(
                curr_good_ops, num_classes=3
            ).float()
            good_operations.append(curr_good_ops)
            good_operations_one_hot.append(curr_good_ops_one_hot)

            curr_bad_func_vecs = torch.stack(curr_bad_func_vecs, dim=0)
            bad_func_vecs.append(curr_bad_func_vecs)
            curr_bad_arg_vecs = torch.stack(curr_bad_arg_vecs, dim=0)
            bad_arg_vecs.append(curr_bad_arg_vecs)
            curr_bad_ops = torch.tensor(curr_bad_ops)
            bad_operations.append(curr_bad_ops)
            curr_dirs = torch.tensor(curr_dirs)

            directions.append(curr_dirs)

        good_func_vecs = torch.stack(good_func_vecs, dim=0)
        good_arg_vecs = torch.stack(good_arg_vecs, dim=0)
        good_operations = torch.stack(good_operations, dim=0).unsqueeze(dim=-1)
        good_operations_one_hot = torch.stack(good_operations_one_hot, dim=0)
        directions = torch.stack(directions, dim=0).unsqueeze(dim=-1)
        good_op_input = torch.cat([good_func_vecs, good_arg_vecs], dim=2)
        good_op_probs = self.operation_model(good_op_input)
        if self.operation_model_type == "mlp":
            #good_op_probs = torch.sigmoid(good_op_probs)
            good_op_probs = torch.softmax(good_op_probs, dim=2)
        good_observed_op_probs = good_op_probs.gather(dim=2, index=good_operations).squeeze(dim=-1)
        good_pred_tree_scores = good_observed_op_probs.prod(dim=1)
#        if verbose:
#            print("forward good_pred_tree_scores:")
#            print(good_pred_tree_scores)

        bad_func_vecs = torch.stack(bad_func_vecs, dim=0)
        bad_arg_vecs = torch.stack(bad_arg_vecs, dim=0)
        bad_operations = torch.stack(bad_operations, dim=0).unsqueeze(dim=-1)
        bad_op_input = torch.cat([bad_func_vecs, bad_arg_vecs], dim=2)
        bad_op_probs = self.operation_model(bad_op_input)
        if self.operation_model_type == "mlp":
            #bad_op_probs = torch.sigmoid(bad_op_probs)
            bad_op_probs = torch.softmax(bad_op_probs, dim=2)
        bad_observed_op_probs = bad_op_probs.gather(dim=2, index=bad_operations).squeeze(dim=-1)
        bad_pred_tree_scores = bad_observed_op_probs.prod(dim=1)
#        if verbose:
#            print("forward bad_pred_tree_scores:")
#            print(bad_pred_tree_scores)

        # TODO use all trees
        #valid = torch.Tensor(valid)[:10]
        valid = torch.Tensor(valid)

        pred_tree_scores = good_pred_tree_scores * bad_pred_tree_scores
        pred_tree_total = pred_tree_scores.sum().item()
#        if verbose:
#            print("forward pred_tree_scores:")
#            print(pred_tree_scores)
#            print("forward pred_tree total:")
#            print(pred_tree_total)
        pred_tree_probs = pred_tree_scores / pred_tree_total

#        if verbose:
#            print("forward pred_tree_probs:")
#            print(pred_tree_probs)

        order_probs = self.ordering_model(good_operations_one_hot)
        if self.ordering_model_type == "mlp":
            order_probs = torch.sigmoid(order_probs)
        
#        hom = hacky_ordering_model(good_operations_one_hot)
#        if verbose:
#            print("forward hom:")
#            print(hom)
#        if verbose:
#            print("forward order probs shape:")
#            print(order_probs.shape)
        # this has the effect of replacing Pr(L | operation) with
        # Pr(R | direction) at nodes where the functor is on the right
        observed_order_probs = order_probs + (directions * (-2*order_probs + 1))
        observed_order_probs = observed_order_probs.squeeze(dim=-1)
        word_order_probs = observed_order_probs.prod(dim=1) * valid
#        if verbose:
#            print("forward word_order_probs:")
#            print(word_order_probs)

        combined_probs = pred_tree_probs * word_order_probs
        #print("combined_probs:")
        #print(combined_probs)
        loss = -1 * torch.log(combined_probs.sum())
        if get_tree_strings:
            return loss, combined_probs, tree_strings
        else:
            return loss, combined_probs

