import numpy as np, random, sys, torch
from copy import deepcopy
from itertools import permutations as perm
from numpy.fft import fft, ifft
from numpy.linalg import norm
from typing import List, Optional

# source for binary tree enumeration: https://algo.monster/liteproblems/894

def compose(v_func, v_arg, op, method="keep_head"):
    """
    Compose functor and argument vectors.
    """
    v_out = None
    if method == "keep_head":
        # argument attachment: head is functor
        if op == "A":
            v_out = v_func
        # argument attachment: head is argument
        elif op == "M":
            v_out = v_arg
        else:
            raise NotImplementedError("composition operation not supported: " + op)

    elif method == "holographic":
        # https://stackoverflow.com/questions/28284257/circular-cross-correlation-python
        # TODO verify that this is the right thing for holographic composition
        v_out = torch.Tensor(ifft(fft(v_func) * fft(v_arg).conj()).real)
    else:
        raise NotImplementedError("composition method not implemented: " + method)
    return v_out


class TreeNode:
    def __init__(self, val="x", op=None, func=None, arg=None):
        self.val = val
        self.op = op
        self.func = func
        self.arg = arg

    def set_leaf_nodes(self, leaf_vals):
        if self.func is None:
            assert self.arg is None
            assert len(leaf_vals) == 1
            self.val = leaf_vals[0]
        else:
            assert self.arg
            stack = [self.func, self.arg]
            while stack:
                leaf_ix = TreeNode._set_leaf_nodes(stack, leaf_vals)
        
    @staticmethod
    def _set_leaf_nodes(stack, leaf_vals):
        curr = stack.pop(0)
        if curr.func is None:
            assert curr.arg is None
            curr.val = leaf_vals.pop(0)
        else:
            assert curr.arg
            stack.insert(0, curr.arg)
            stack.insert(0, curr.func)

    def __repr__(self):
        if self.func:
            assert self.arg
            assert self.op
            return "(%s %s)%s" % (self.func, self.arg, self.op)
        else:
            return str(self.val)

    def __eq__(self, other):
        if self.val != other.val: return False
        if self.func:
            if not other.func: return False
            else: return self.func == other.func and self.arg == other.arg
        elif other.func: return False
        else: return True


class Tree:
    def __init__(self, node, vectors):
        self.root = node
        self.annotate(vectors)
        self.max_func_chain = max(
            n.func_chain for n in self.nonterminals
        )

    def annotate(self, vectors):
        # depths (all)
        # list of leaves
        # list of nonterminals
        self._annotate_functor_chains()
        # index ranges (all)
        # separability (branching)
        # if separable, order (branching)
        self._annotate_separability()
        # vectors (leaves)
        # vectors (branching) (post order DFS)
        self._annotate_vectors(vectors)

    # this is keeps track of which predicate role should be considered
    # for argument attachment
    def _annotate_functor_chains(self):
        self.root.func_chain = 1
        self.leaves = list()
        self.nonterminals = list()
        stack = [self.root]
        while len(stack) > 0:
            curr = stack.pop(0)
            curr_op = curr.op
            curr_func_chain = curr.func_chain
            if curr.func:
                assert curr.arg
                curr.terminal = False
                self.nonterminals.append(curr)
                if curr_op == "A":
                    curr.arg.func_chain = 1
                    curr.func.func_chain = curr_func_chain + 1
                elif curr_op == "M":
                    curr.arg.func_chain = curr_func_chain
                    curr.func.func_chain = 1
                else:
                    raise NotImplementedError("composition operation not supported: " + curr_op)
                stack.insert(0, curr.arg)
                stack.insert(0, curr.func)
            else:
                curr.terminal = True
                self.leaves.append(curr)
        
    def _annotate_separability(self):
        Tree._annotate_node_separability(self.root)

    @staticmethod
    def _annotate_node_separability(node):
        if node.terminal:
            node.separable = True
            node.min_ix = node.val
            node.max_ix = node.val
            node.functor_position = "?"
        else:
            Tree._annotate_node_separability(node.func)
            Tree._annotate_node_separability(node.arg)
            node.separable = False
            node.min_ix = "X" 
            node.max_ix =  "X"
            node.functor_position = "X"
            if node.func.separable and node.arg.separable:
                if node.func.max_ix + 1 == node.arg.min_ix:
                    node.separable = True
                    node.min_ix = node.func.min_ix
                    node.max_ix = node.arg.max_ix
                    node.functor_position = "L"
                elif node.arg.max_ix+1 == node.func.min_ix:
                    node.separable = True
                    node.min_ix = node.arg.min_ix
                    node.max_ix = node.func.max_ix
                    node.functor_position = "R"


    def _annotate_vectors(self, vectors):
        for leaf in self.leaves:
            leaf.vector = vectors[leaf.val]
        Tree._annotate_node_vector(self.root)

    @staticmethod
    def _annotate_node_vector(node):
        if not node.terminal:
            #print("nt pred:", node)
            Tree._annotate_node_vector(node.func)
            Tree._annotate_node_vector(node.arg)
            # TODO define more general composition
            node.vector = compose(node.func.vector, node.arg.vector, node.op)

    def __repr__(self):
        rep = ""
        stack = [(self.root, 0)]
        while len(stack) > 0:
            curr, depth = stack.pop(0)
            rep += "  "*depth
            #rep += "val:" + str(curr.val) + " "
            rep += "fchain:" + str(curr.func_chain) + " "
            rep += "fpos:" + str(curr.functor_position) + " "
            rep += "vec:" + str(curr.vector) + " "
            rep += "rng:" + "{}~{}".format(curr.min_ix, curr.max_ix) + "\n"
            if curr.func:
                assert curr.arg
                stack.insert(0, (curr.arg, depth+1))
                stack.insert(0, (curr.func, depth+1))
        #return str(self.root)
        return rep[:-1]


def full_binary_trees(total_nodes: int) -> List[Optional[TreeNode]]:
    # If there is only one node, return list containing a single TreeNode.
    if total_nodes == 1:
        return [TreeNode()]
  
    # List to store all unique FBTs created from 'total_nodes' nodes.
    fbts = []
  
    # Iterate over the number of nodes left after one is taken as the current root.
    for nodes_in_left_subtree in range(total_nodes - 1):
        nodes_in_right_subtree = total_nodes - 1 - nodes_in_left_subtree
      
        # Generate all full binary trees for the number of nodes in left subtree.
        left_subtrees = full_binary_trees(nodes_in_left_subtree)
        # Generate all full binary trees for the number of nodes in right subtree.
        right_subtrees = full_binary_trees(nodes_in_right_subtree)

        # Combine each left subtree with each right subtree and add the current node as root.
        for left in left_subtrees:
            for right in right_subtrees:
                fbts.append(TreeNode(op="A", func=left, arg=right))
                fbts.append(TreeNode(op="M", func=left, arg=right))
  
    # Return the list of all unique full binary trees.
    return fbts


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
    