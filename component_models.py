import torch
from torch import nn

QUASI_INF = 1e9
DEBUG = False

def printDebug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# words:
# 0: happy
# 1: people
# 2: eat
# 3: yummy
# 4: donuts

class CoocOpModel(nn.Module):
    def __init__(self, cooccurrences):
        super(CoocOpModel, self).__init__()
        self.cooccurrences = cooccurrences

    # TODO double check that these multiplications are done correctly
    def forward(self, func, arg):
        # dim: ...
        single_op_mask = torch.full(func.shape[:-1], -QUASI_INF)
        # dim: ...
        # if this flag is 1, functor has already combined with an arg1
        arg1_flag = func[..., -2]
        # dim: ...
        # if this flag is 1, functor has already combined with an arg2
        arg2_flag = func[..., -1]
        # dim: ...
        no_arg1 = arg1_flag * single_op_mask
        # block arg2 if either flag is 1
        # dim: ...
        no_arg2 = (1 - (1-arg1_flag)*(1-arg2_flag)) * single_op_mask
        # don't block modification
        no_mod = 0 * single_op_mask
        # shape: ... x ops
        op_mask = torch.stack([no_arg1, no_arg2, no_mod], dim=-1)

        # dim: ... x dvec
        func_vecs = func[..., :-2]
        # dim: ... x 1 x dvec x 1
        func_vecs = func_vecs.unsqueeze(-2).unsqueeze(-1)
        # shape of self.cooccurrences: dvec x dvec x ops
        # dim: ops x dvec x dvec
        cooc = self.cooccurrences.permute(2,0,1)
        # dim: ... x ops x dvec
        cooc = (cooc * func_vecs).sum(dim=-2)
        # dim: ... x dvec
        # flags on argument vector don't matter
        arg_vecs = arg[..., :-2]
        # dim: ... x 1 x dvec
        arg_vecs = arg_vecs.unsqueeze(-2)
        # dim: ... x ops
        cooc = (cooc * arg_vecs).sum(dim=-1)
        printDebug("op_mask:")
        printDebug(op_mask)
        printDebug("combined:")
        printDebug(cooc + op_mask)
        return cooc + op_mask


#        func = func_and_arg[..., :self.dvec]
#        arg = func_and_arg[..., self.dvec:]
#        # NOTE: the einsums seem to be very slow
#        # NOTE: for some reason, this operation model seems to cause much
#        # smaller gradient updates to the word embeddings
#        # func dim: trees x nts x dvec
#        # COOC_FIVE_WORD dim: dvec x dvec x ops
#        # output dim: trees x nts x dvec x ops
#        # TODO pass cooccurrences in rather than using hard-coded values
#        cooc = torch.einsum("tnu,uvo->tnvo", func, COOC_FIVE_WORD)
#        # arg dim: trees x nts x dvec
#        # cooc dim: trees x nts x dvec x ops
#        # output dim: trees x nts x ops
#        cooc = torch.einsum("tnv,tnvo->tno", arg, cooc)
#        return cooc
