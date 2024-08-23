import torch

# words:
# 0: happy
# 1: people
# 2: eat
# 3: yummy
# 4: donuts

# operations:
# 0: arg 1
# 1: arg 2
# 2: modification
# 3: none

# COOC_HAPPY[w, op] gives Pr(op | functor=happy, argument=w)
COOC_HAPPY = torch.Tensor([
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1]
])
#COOC_HAPPY = torch.Tensor([
#    [0.25, 0.25, 0.25, 0.25],
#    [0.25, 0.25, 0.25, 0.25],
#    [0.25, 0.25, 0.25, 0.25],
#    [0.25, 0.25, 0.25, 0.25],
#    [0.25, 0.25, 0.25, 0.25]
#])

COOC_PEOPLE = torch.Tensor([
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1]
])
#COOC_PEOPLE = torch.Tensor([
#    [0.25, 0.25, 0.25, 0.25],
#    [0.25, 0.25, 0.25, 0.25],
#    [0.25, 0.25, 0.25, 0.25],
#    [0.25, 0.25, 0.25, 0.25],
#    [0.25, 0.25, 0.25, 0.25]
#])

COOC_EAT = torch.Tensor([
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 1, 0, 0]
])
#COOC_EAT = torch.Tensor([
#    [0.25, 0.25, 0.25, 0.25],
#    [0.25, 0.25, 0.25, 0.25],
#    [0.25, 0.25, 0.25, 0.25],
#    [0.25, 0.25, 0.25, 0.25],
#    [0.25, 0.25, 0.25, 0.25]
#])

COOC_YUMMY = torch.Tensor([
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
])
#COOC_YUMMY = torch.Tensor([
#    [0.25, 0.25, 0.25, 0.25],
#    [0.25, 0.25, 0.25, 0.25],
#    [0.25, 0.25, 0.25, 0.25],
#    [0.25, 0.25, 0.25, 0.25],
#    [0.25, 0.25, 0.25, 0.25]
#])

COOC_DONUTS = torch.Tensor([
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 1]
])
#COOC_DONUTS = torch.Tensor([
#    [0.25, 0.25, 0.25, 0.25],
#    [0.25, 0.25, 0.25, 0.25],
#    [0.25, 0.25, 0.25, 0.25],
#    [0.25, 0.25, 0.25, 0.25],
#    [0.25, 0.25, 0.25, 0.25]
#])

COOC_FIVE_WORD = torch.stack([COOC_HAPPY, COOC_PEOPLE, COOC_EAT, COOC_YUMMY, COOC_DONUTS], dim=0)

def fixed_op_three_word(func_and_arg):
    hacky_output = list()
    """
    non-learnable operation mlp for proof-of-concept system
    in test_inducer.py
    possible operations: arg1 attachment, arg2 attachment, modification, none
    input dim: trees x nonterminals x (2*dvec)
    output dim: trees x nonterminals x 4
    """
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
    return torch.Tensor(hacky_output)


def cooc_op_five_word(func_and_arg):
    """
    non-learnable operation mlp for proof-of-concept system
    in test_inducer.py
    possible operations: arg1 attachment, arg2 attachment, modification, none
    input dim: trees x nonterminals x (2*dvec)
    output dim: trees x nonterminals x 4
    """
    print("CEC input shape:")
    print(func_and_arg.shape)
    # TODO do this with tensor operations so it can backprop
    hacky_output = list()
    for tree in func_and_arg:
        t_output = list()
        for nont in tree:
            # TODO correct this to do matrix multiplication (so that vfunc and varg can be distributions, not just one-hot)
            func = nont[:5].unsqueeze(0)
            arg = nont[5:].unsqueeze(0)
            # func dim: 1x5
            # COOC_FIVE_WORD dim: 5x5x4
            # output dim after squeeze: 5x4
            cooc = torch.einsum("ij,jkl->ikl", func, COOC_FIVE_WORD).squeeze(0)
            # arg dim: 1x5
            # output dim after squeeze: 4
            cooc = torch.einsum("ij,jk->ik", arg, cooc).squeeze(0)
            t_output.append(list(cooc))
            #ix_func = torch.argmax(nont[:5])
            #ix_arg = torch.argmax(nont[5:])
            #t_output.append(list(COOC_FIVE_WORD[ix_func, ix_arg]))
        hacky_output.append(t_output)
    return torch.Tensor(hacky_output)


def hacky_operation_model(func_and_arg, version="cooc_four_word"):
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