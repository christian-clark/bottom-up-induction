import numpy as np, random, sys
from copy import deepcopy
from itertools import permutations as perm
from numpy.fft import fft, ifft
from numpy.linalg import norm
from typing import List, Optional

# source for binary tree enumeration: https://algo.monster/liteproblems/894

# https://stackoverflow.com/questions/28284257/circular-cross-correlation-python
# TODO verify that this is the right composition function
def periodic_corr(x, y):
    """Periodic correlation, implemented using the FFT.

    x and y must be real sequences with the same length.
    """
    return ifft(fft(x) * fft(y).conj()).real


def composition_prob(a, b, role=1):
    # cosine similarity as a placeholder
    # TODO make this a learnable function
    prob = 0
    if role == 1:
        prob = np.dot(a, b)/(norm(a)*norm(b))
    elif role == 2:
        prob = np.dot(a, b)/(norm(a)*norm(b))
    return prob


def word_order_prob(direction, role=1):
    # TODO make these probabilities learnable (don't hardcode)
    assert direction in ["L", "R"]
    prob = 0.5
    if role == 1:
        if direction == "L":
            prob = OP_A_ROLE_1_LEFT_PROB
        else:
            prob = 1 - OP_A_ROLE_1_LEFT_PROB
    elif role == 2:
        if direction == "L":
            prob = OP_A_ROLE_2_LEFT_PROB
        else:
            prob = 1 - OP_A_ROLE_2_LEFT_PROB
    return prob


#def set_leaf_nodes(tree, leaf_vals):
#    if tree.left == None:
#        assert tree.right == None
#        assert len(leaf_vals) == 1
#        tree.val = leaf_vals[0]
#    else:
#        assert tree.right
#        stack = [tree.left, tree.right]
#        while stack:
#            leaf_ix = _set_leaf_nodes(stack, leaf_vals)
#    
#        
#def _set_leaf_nodes(stack, leaf_vals):
#    curr = stack.pop(0)
#    if curr.left == None:
#        assert curr.right == None
#        curr.val = leaf_vals.pop(0)
#    else:
#        assert curr.right
#        stack.insert(0, curr.right)
#        stack.insert(0, curr.left)
    

class TreeNode:
    def __init__(self, val="x", left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def set_leaf_nodes(self, leaf_vals):
        if self.left == None:
            assert self.right == None
            assert len(leaf_vals) == 1
            self.val = leaf_vals[0]
        else:
            assert self.right
            stack = [self.left, self.right]
            while stack:
                leaf_ix = TreeNode._set_leaf_nodes(stack, leaf_vals)
        
    @staticmethod
    def _set_leaf_nodes(stack, leaf_vals):
        curr = stack.pop(0)
        if curr.left == None:
            assert curr.right == None
            curr.val = leaf_vals.pop(0)
        else:
            assert curr.right
            stack.insert(0, curr.right)
            stack.insert(0, curr.left)

    def __repr__(self):
        if self.left:
            assert self.right
            return "(%s %s)" % (self.left, self.right)
        else:
            return str(self.val)


class Tree:
    def __init__(self, node, vectors):
        self.root = node
        self.annotate(vectors)

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

    def unnormalized_score(self):
        # TODO normalize this by summing pred_tree_score across
        # all possible trees
        return self.root.pred_tree_score * self.root.word_order_prob

    # also annotates whether nodes are terminal and makes lists of
    # all the terminal and nonterminal nodes
    # TODO depth isn't quite the right thing -- need to annotate # functor
    # connections from root (or a nonfunctor branch) to the current node
    # in the predicate tree this is count of left children leading to the
    # current node since a right child has intervened
#    def _annotate_depth(self):
#        self.root.depth = 0
#        self.leaves = list()
#        self.nonterminals = list()
#        stack = [self.root]
#        while len(stack) > 0:
#            curr = stack.pop(0)
#            curr_depth = curr.depth
#            if curr.left:
#                assert curr.right
#                curr.terminal = False
#                self.nonterminals.append(curr)
#                curr.right.depth = curr_depth + 1
#                curr.left.depth = curr_depth + 1
#                stack.insert(0, curr.right)
#                stack.insert(0, curr.left)
#            else:
#                curr.terminal = True
#                self.leaves.append(curr)


    def _annotate_functor_chains(self):
        self.root.func_chain = 1
        self.leaves = list()
        self.nonterminals = list()
        stack = [self.root]
        while len(stack) > 0:
            curr = stack.pop(0)
            curr_func_chain = curr.func_chain
            if curr.left:
                assert curr.right
                curr.terminal = False
                self.nonterminals.append(curr)
                curr.right.func_chain = 1
                curr.left.func_chain = curr_func_chain + 1
                stack.insert(0, curr.right)
                stack.insert(0, curr.left)
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
            Tree._annotate_node_separability(node.left)
            Tree._annotate_node_separability(node.right)
            node.separable = False
            node.min_ix = "X" 
            node.max_ix =  "X"
            node.functor_position = "X"
            if node.left.separable and node.right.separable:
                if node.left.max_ix + 1 == node.right.min_ix:
                    node.separable = True
                    node.min_ix = node.left.min_ix
                    node.max_ix = node.right.max_ix
                    node.functor_position = "L"
                elif node.right.max_ix+1 == node.left.min_ix:
                    node.separable = True
                    node.min_ix = node.right.min_ix
                    node.max_ix = node.left.max_ix
                    node.functor_position = "R"


    def _annotate_vectors(self, vectors):
        for leaf in self.leaves:
            leaf.vector = vectors[leaf.val]
            leaf.pred_tree_score = 1
            leaf.word_order_prob = 1
        #for nt in self.nonterminals:
        #    nt.vector = -1
        Tree._annotate_node_vector(self.root)


    @staticmethod
    def _annotate_node_vector(node):
        if not node.terminal:
            #print("nt pred:", node)
            Tree._annotate_node_vector(node.left)
            Tree._annotate_node_vector(node.right)
            # note: it is a convention in the argument structure trees that
            # functor is always the left child and the argument the right child
            l_vec = node.left.vector
            r_vec = node.right.vector
            node.vector = periodic_corr(l_vec, r_vec)
            l_pred_tree_score = node.left.pred_tree_score
            r_pred_tree_score = node.right.pred_tree_score
            # in predicate-argument tree, left child is functor and right
            # is argument. order matters for composition prob
            curr_score = composition_prob(l_vec, r_vec, role=node.func_chain)
            #print("l score:", l_pred_tree_score)
            #print("r score:", r_pred_tree_score)
            #print("curr score:", curr_score)
            node.pred_tree_score = \
                l_pred_tree_score * r_pred_tree_score * curr_score
            l_word_order_prob = node.left.word_order_prob
            r_word_order_prob = node.right.word_order_prob
            if node.separable:
                curr_prob = word_order_prob(
                    node.functor_position, role=node.func_chain
                )
                node.word_order_prob = \
                    l_word_order_prob * r_word_order_prob * curr_prob
            else:
                node.word_order_prob = 0


    def __repr__(self):
        rep = ""
        stack = [(self.root, 0)]
        while len(stack) > 0:
            curr, depth = stack.pop(0)
            rep += "  "*depth
            #rep += "val:" + str(curr.val) + " "
            rep += "fchain:" + str(curr.func_chain) + " "
            rep += "fpos:" + str(curr.functor_position) + " "
            rep += "pts:" + str(round(curr.pred_tree_score, 6)) + " "
            rep += "wop:" + "{:4.2f}".format(curr.word_order_prob) + " "
            rep += "vec:" + str(curr.vector) + " "
            rep += "rng:" + "{}~{}".format(curr.min_ix, curr.max_ix) + "\n"
            if curr.left:
                assert curr.right
                stack.insert(0, (curr.right, depth+1))
                stack.insert(0, (curr.left, depth+1))
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
                fbts.append(TreeNode("x", left, right))
  
    # Return the list of all unique full binary trees.
    return fbts


if __name__ == "__main__":
    SENT_LEN = int(sys.argv[1])
    VECTOR_DIM = 4
    # TODO learn these probabilities
    OP_A_ROLE_1_LEFT_PROB = 0.33
    OP_A_ROLE_2_LEFT_PROB = 0.8

    # bigger values will be extremely time-consuming to enumerate!
    assert SENT_LEN <= 6

    # generate random vectors
    vectors = list()
    for i in range(SENT_LEN):
        v = list()
        for j in range(VECTOR_DIM):
            r = random.random()
            # scale between -1 and 1
            #r = r*2 - 1
            v.append(r)
        vectors.append(np.array(v))

    # generate random operation probabilities
    # TODO take role into account (A1 and A2 should be different ops)
    prob_l_func = random.random()
    prob_r_func = 1 - prob_l_func

    #print(opA_prob(vectors[0], vectors[1]))
    # iterate through possible trees

    np.set_printoptions(formatter={'float': '{:0.2f}'.format})
    raw_trees = full_binary_trees(2*SENT_LEN - 1)
    trees = list()
    tree_count = 0
    total_score = 0
    for r in raw_trees:
        ps = perm(range(SENT_LEN))
        for p in ps:
            rcurr = deepcopy(r)
            rcurr.set_leaf_nodes(list(p))
            t = Tree(rcurr, vectors)
            trees.append(t)
            total_score += t.unnormalized_score()
            tree_count += 1

    for t in sorted(trees, key=lambda t:t.unnormalized_score()):
        print(t.root)
        print(t)
        print("probability:", t.unnormalized_score()/total_score)
        print("============")
    print("TOTAL TREES:", tree_count)
    #trees = full_binary_trees(7)
    #t = trees[0]
    #print(t)
    #set_leaf_nodes(t, [5, 6, 7, 8])
    #set_leaf_nodes(t, [4, 6, 2, 4])
    #print(t)
