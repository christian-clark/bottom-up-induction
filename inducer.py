import torch
from copy import deepcopy
from itertools import permutations as perm
from torch import nn

from component_models import CoocOpModel
from tree import enumerate_trees

MAX_ROLE = 2
TINY = 1e-9
DEBUG = False

def printDebug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

class Inducer(nn.Module):
    def __init__(self, config, learn_vectors, fixed_vectors, cooccurrences=None):
        super(Inducer, self).__init__()
        self.learn_vectors = learn_vectors
        self.fixed_vectors = fixed_vectors
        self.cooccurrences = cooccurrences
        self.vocab_size = fixed_vectors.shape[0]
        self.dvec = fixed_vectors.shape[1]
        self.emb = torch.nn.Embedding(self.vocab_size, self.dvec)
        # TODO read this from config once other composition methods are
        # supported
        self.composition_method = "head"
        self.beam_size = config.getint("beam_size")
        # operation probabilities
        # - input: functor and argument vectors (dim: 2*d)
        # - output: probability of composition operations:
        #       arg1, arg2, modification, or noop
        # TODO convert string directly into function?
        self.operation_model_type = config["operation_model_type"]
        if self.operation_model_type == "cooc_op":
            self.operation_model = CoocOpModel(cooccurrences)
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
    
    def compose_vectors(self, func, arg, op):
        # func dim: ... x dvec x n
        # arg dim: ... x dvec x n
        # op dim: ... x 1 x n
        # n is the total number of (func, arg, op) triplets to compose
        if self.composition_method == "head":
            is_arg1 = (op == 0)
            is_arg2 = (op == 1)
            is_mod = (op == 2)
            # in arg1 or arg2 attachment, functor propagates
            prop_func = is_arg1 | is_arg2
            # in modifier attachment, argument propagates
            prop_arg = is_mod

            # dim: ... x 1 x n
            func_arg1_flag = func[..., -2:-1, :].long()
            # dim: ... x 1 x n
            new_arg1_flag = is_arg1 | func_arg1_flag
            # dim: ... x 1 x n
            func_arg2_flag = func[..., -1:, :].long()
            # dim: ... x 1 x n
            new_arg2_flag = is_arg2 | func_arg2_flag

            printDebug("new_arg1_flag:", new_arg1_flag)
            printDebug("new_arg2_flag:", new_arg2_flag)

            func[..., -2:-1, :] = new_arg1_flag
            func[..., -1:, :] = new_arg2_flag

            # check that all ops are acceptable values
            assert all((prop_arg | prop_func).reshape(-1)), "all operations must be 0, 1, or 2"
            output = prop_func * func + prop_arg * arg
        else:
            raise NotImplementedError
        return output

    def forward(self, x, return_backpointers=False):
        """x is an n x d vector containing the d-dimensional vectors for a
        sentence of length n"""
        torch.autograd.set_detect_anomaly(True)
        # dim: imax x dvec
        x = self.vectorize_sentence(x)
        if len(x) == 1:
            raise Exception("single-word sentences not supported")
        sent_len = x.shape[0]
        beam = self.beam_size
        # vectors are concatenated with two extra bits that indicate whether
        # a word has combined with a first and/or second argument already
        dvec = self.dvec + 2
        # dim: ijdiff x imax x beam x dvec
        # and/or second argument already
        left_chart_vecs = torch.zeros((sent_len, sent_len, beam, dvec))
        # dim: ijdiff x imax x beam
        #left_chart_op_scores = torch.zeros((sent_len, sent_len, beam))
        # can't set to zeros because this gets log transformed
        left_chart_op_scores = torch.full((sent_len, sent_len, beam), fill_value=TINY)
        # dim: ijdiff x imax x beam
        left_chart_ord_probs = torch.zeros((sent_len, sent_len, beam))
        #left_chart_vecs[0, :, 0] = x
        left_chart_vecs[0, :, 0, :self.dvec] = x
        left_chart_op_scores[0, :, 0] = 1
        left_chart_ord_probs[0, :, 0] = 1
        if return_backpointers:
            # dim: ijdiff x imax x beam x 5
            # 5 is for storing ijdiff, left beam ix, right beam ix,
            # dir and op
            backpointers = torch.full((sent_len, sent_len, beam, 5), fill_value=-1)
            #backpointers = [[[None for i in range(beam)] for j in range(sent_len)] for k in range(sent_len)]

        right_chart_vecs = left_chart_vecs
        right_chart_op_scores = left_chart_op_scores
        right_chart_ord_probs = left_chart_ord_probs
        for ij_diff in range(1, sent_len):
            printDebug("== ijdiff: {} ==".format(ij_diff))
        #for ij_diff in range(1, 2):
            imin = 0
            imax = sent_len - ij_diff
            jmin = ij_diff
            jmax = sent_len
            height = ij_diff

            ########
            # get operation scores from combining all pairs of vectors in left
            # and right beams
            ########
            # dim: ijdiff x imax x beam x dvec
            b_vec = left_chart_vecs[0:height, imin:imax]
            # dim: ijdiff x imax x lbeam x rbeam x dvec
            b_vec = b_vec.unsqueeze(dim=-2).expand(-1, -1, -1, beam, -1)
            # dim: ijdiff x imax x (lbeam * rbeam) x dvec
            b_vec = b_vec.reshape(ij_diff, imax, beam**2, dvec)

            printDebug("current b_vec:")
            printDebug(b_vec.reshape(ij_diff, imax, beam, beam, dvec))

            # dim: ijdiff x imax x beam x dvec
            c_vec = torch.flip(right_chart_vecs[0:height, jmin:jmax], dims=[0])
            # dim: ijdiff x imax x lbeam x rbeam x dvec
            c_vec = c_vec.unsqueeze(dim=-3).expand(-1, -1, beam, -1, -1)
            # dim: ijdiff x imax x (lbeam * rbeam) x dvec
            c_vec = c_vec.reshape(ij_diff, imax, beam**2, dvec)

            printDebug("current c_vec:")
            printDebug(c_vec.reshape(ij_diff, imax, beam, beam, dvec))

            # scores when left child is functor and right is argument
            # dim: ijdiff x imax x (lbeam * rbeam) x op
            bc_op_scores = self.operation_model(b_vec, c_vec)
            # scores when right child is functor and left is argument
            # dim: ijdiff x imax x (lbeam * rbeam) x op
            cb_op_scores = self.operation_model(c_vec, b_vec)
            # dim: ijdiff x imax x (lbeam * rbeam) x (dir * op)
            new_op_scores = torch.cat([bc_op_scores, cb_op_scores], dim=-1)
            # all (functor, argument) pairs and operations along the same dimension
            # dim: ijdiff x imax x (lbeam * rbeam * dir * op)
            new_op_scores = new_op_scores.reshape(ij_diff, imax, beam**2 * 6)
            # new_op_scores is the score for the root of the subtree. need to
            # combine with scores from its two children

            printDebug("new_op_scores:")
            printDebug(new_op_scores.reshape(ij_diff, imax, beam, beam, 6))

            ########
            # get parent vectors from children, based no what operations happen
            ########
            # dim: ijdiff x imax x (lbeam * rbeam) x dvec x op
            b_vec = b_vec.unsqueeze(-1).repeat(1, 1, 1, 1, 3)
            c_vec = c_vec.unsqueeze(-1).repeat(1, 1, 1, 1, 3)
            # dim: op
            one_hot_op = torch.arange(0, 3)
            # dim = 1 x 1 x 1 x 1 x op
            one_hot_op = one_hot_op.reshape(1, 1, 1, 1, 3)
            # dim: ijdiff x imax x (lbeam * rbeam) x dvec x op
            bc_vecs = self.compose_vectors(b_vec, c_vec, one_hot_op)
            # dim: ijdiff x imax x (lbeam * rbeam) x dvec x op
            cb_vecs = self.compose_vectors(c_vec, b_vec, one_hot_op)
            # dim: ijdiff x imax x (lbeam * rbeam) x dvec x (dir * op)
            new_vecs = torch.cat([bc_vecs, cb_vecs], dim=-1)
            # dim: max x ijdiff x (lbeam * rbeam) x (dir * op) x dvec
            new_vecs = new_vecs.permute(1, 0, 2, 4, 3)
            # dim: imax x (ijdiff * lbeam * rbeam * dir * op) x dvec
            new_vecs = new_vecs.reshape(imax, -1, dvec)

            printDebug("new_vecs:")
            printDebug(new_vecs.reshape(ij_diff, imax, beam, beam, 6, dvec))

            ########
            # get combined operation scores from already formed subtrees
            ########
            # shape: ijdiff x imax x beam
            # not sure why clone is necessary, but torch complains otherwise
            b_op = left_chart_op_scores[0:height, imin:imax].clone()
            # dim: ijdiff x imax x beam 
            b_op = torch.log(b_op)
            b_factor = torch.arange(ij_diff)#.unsqueeze(-1).unsqueeze(-1)
            # there should be only one possible beam element for the leaf
            # this hack ensures that other items in the leaf's beam won't be
            # chosen
            b_factor[0] = 1
            b_factor = b_factor.unsqueeze(-1).unsqueeze(-1)
            # dim: ijdiff x imax x beam 
            b_op = b_op * b_factor

            printDebug("current unnormalized b_op:")
            printDebug(b_op.reshape(ij_diff, imax, beam))

            # shape: ijdiff x imax x beam 
            c_op = torch.flip(right_chart_op_scores[0:height, jmin:jmax], dims=[0])
            # dim: ijdiff x imax x beam 
            c_op = torch.log(c_op)
            c_factor = torch.arange(ij_diff)#.unsqueeze(-1).unsqueeze(-1)
            # there should be only one possible beam element for the leaf
            # this hack ensures that other items in the leaf's beam won't be
            # chosen
            c_factor[0] = 1
            c_factor = torch.flip(c_factor, dims=[0])
            c_factor = c_factor.unsqueeze(-1).unsqueeze(-1)
            # dim: ijdiff x imax x beam 
            c_op = c_op * c_factor
            #c_op = torch.log(c_op)

            printDebug("current unnormalized c_op:")
            printDebug(c_op.reshape(ij_diff, imax, beam))

            # dim: ijdiff x imax x lbeam x rbeam
            old_op_scores = b_op[..., None] + c_op[..., None, :]
            # dim: ijdiff x imax x lbeam x rbeam
            old_op_scores = old_op_scores.reshape(ij_diff, imax, beam**2)
            # repeat to make it line up with new_op_scores's different operations and directions
            # dim: ijdiff x imax x (lbeam * rbeam * dir * op)
            old_op_scores = old_op_scores.unsqueeze(-1).expand(-1, -1, -1, 6).reshape(ij_diff, imax, -1)

            printDebug("current old_op_scores:")
            printDebug(old_op_scores.reshape(ij_diff, imax, beam, beam, 6))

            ########
            ## combine operation scores at the root of the new subtree with
            ## scores from the already formed subtrees
            ########
            # dim: ijdiff x imax x (lbeam * rbeam * dir * op)
            combined_op_scores = new_op_scores + old_op_scores
            # ijdiff is the same as the number of nonterminal nodes in the partial tree so far
            combined_op_scores = torch.exp(combined_op_scores/ij_diff)
            printDebug("combined_op_scores:")
            printDebug(combined_op_scores.reshape(ij_diff, imax, beam, beam, 6))
            # dim: imax x (ijdiff * lbeam * rbeam * dir * op)
            # the last dim has all the possibilities to find the top k over
            combined_op_scores = combined_op_scores.permute(1, 0, 2).reshape(imax, -1)

            ########
            # get probs of ordering the two child subtrees functor left, arg
            # right or arg left, functor right
            ########
            # dim: op
            one_hot_ordering = nn.functional.one_hot(torch.arange(3)).float()
            # dim: op
            l_ord_probs = torch.sigmoid(self.ordering_model(one_hot_ordering)).squeeze(dim=1)
            # dim: op
            r_ord_probs = 1 - l_ord_probs
            # dim: (dir * op)
            new_ord_probs = torch.cat([l_ord_probs, r_ord_probs], dim=0)
            # dim: ijdiff x imax x (lbeam * rbeam) x (dir * op)
            new_ord_probs = new_ord_probs.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(ij_diff, imax, beam**2, -1)
            # dim: ijdiff x imax x (lbeam * rbeam * dir * op)
            new_ord_probs = new_ord_probs.reshape(ij_diff, imax, -1)
            printDebug("new_ord_probs:")
            printDebug(new_ord_probs.reshape(ij_diff, imax, beam, beam, 6))


            ########
            # get ordering probabilities from already formed subtrees
            ########
            # dim: ijdiff x imax x beam 
            # not sure why clone is necessary, but torch complains otherwise
            b_ord = left_chart_ord_probs[0:height, imin:imax].clone()
            # dim: ijdiff x imax x beam 
            c_ord = torch.flip(right_chart_ord_probs[0:height, jmin:jmax], dims=[0])
            # dim: ijdiff x imax x beam x beam
            old_ord_probs = b_ord[..., None] * c_ord[..., None, :]
            # dim: ijdiff x imax x (lbeam * rbeam)
            old_ord_probs = old_ord_probs.reshape(ij_diff, imax, beam**2)
            # dim: ijdiff x imax x (lbeam * rbeam * dir * op)
            old_ord_probs = old_ord_probs.unsqueeze(-1).expand(-1, -1, -1, 6).reshape(ij_diff, imax, -1)
            printDebug("old_ord_probs:")
            printDebug(old_ord_probs.reshape(ij_diff, imax, beam, beam, 6))

            ########
            # combine ordinerg probs at the root of the new subtree with
            # probs from the already formed subtrees
            ########
            combined_ord_probs = new_ord_probs * old_ord_probs
            printDebug("combined_ord_probs:")
            printDebug(combined_ord_probs.reshape(ij_diff, imax, beam, beam, 6))
            # dim: imax x (ijdiff * lbeam * rbeam * dir * op)
            combined_ord_probs = combined_ord_probs.permute(1, 0, 2).reshape(imax, -1)

            ########
            # combine operation and ordering scores, and select top k items
            # for next step's beam
            ########
            # dim: imax x (ijdiff * lbeam * rbeam * dir * op)
            combined_full_scores = combined_op_scores * combined_ord_probs
            # dim: imax x beam
            top_k = torch.topk(combined_full_scores, k=beam, dim=-1)
            top_ixs = top_k.indices

            if return_backpointers:
                top_ops = top_ixs % 3
                printDebug("top_ops:")
                printDebug(top_ops)
                top = top_ixs // 3
                top_dirs = top % 2
                printDebug("top_dirs:")
                printDebug(top_dirs)
                top = top // 2
                top_rbeam_ix = top % beam
                printDebug("top_rbeam_ix:")
                printDebug(top_rbeam_ix)
                top = top // beam
                top_lbeam_ix = top % beam
                printDebug("top_lbeam_ix:")
                printDebug(top_lbeam_ix)
                top_ijdiff = (top // beam)
                printDebug("top_ijdiff:")
                printDebug(top_ijdiff)
                # dim: imax x beam x 5
                bp = torch.stack([top_ijdiff, top_lbeam_ix, top_rbeam_ix, top_dirs, top_ops], dim=2)
                backpointers[ij_diff,imin:imax] = bp

            # dim: imax x beam
            selected_op_scores = combined_op_scores.gather(dim=-1, index=top_ixs)
            left_chart_op_scores[height, imin:imax] = selected_op_scores
            right_chart_op_scores[height, jmin:jmax] = selected_op_scores

            # dim: imax x beam
            selected_ord_probs = combined_ord_probs.gather(dim=-1, index=top_ixs)
            left_chart_ord_probs[height, imin:imax] = selected_ord_probs
            right_chart_ord_probs[height, jmin:jmax] = selected_ord_probs

            # dim: imax x beam x dvec
            top_ixs_vec = top_ixs.unsqueeze(dim=-1).expand(-1, -1, dvec)
            # dim: imax x beam x dvec
            selected_vecs = new_vecs.gather(dim=-2, index=top_ixs_vec)
            left_chart_vecs[height, imin:imax] = selected_vecs
            right_chart_vecs[height, jmin:jmax] = selected_vecs

#        printDebug("left_chart_op_scores:")
#        printDebug(left_chart_op_scores)
#        printDebug("left_chart_ord_probs:")
#        printDebug(left_chart_ord_probs)

        top_node_op_scores = left_chart_op_scores[sent_len-1, 0]
        top_node_ord_probs = left_chart_ord_probs[sent_len-1, 0]
        top_node_scores = top_node_op_scores * top_node_ord_probs
        loss = -1 * torch.sum(top_node_scores, dim=0)
        if return_backpointers:
            return loss, top_node_scores, backpointers
        else:
            return loss, top_node_scores
    