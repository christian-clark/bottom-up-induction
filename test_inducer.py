import torch
from inducer import Inducer

# features: animate, eating, edible
# words: people, eat, donuts

#role1(func, arg):
#func[eating] AND arg[animate]

#role2(func, arg):
#func[eating] AND arg[edible]

#composition: just keep functor vector

x = torch.Tensor([
    [1, 0.5, 0],
    [0.5, 1, 0.5],
    [0, 0.5, 1]
])

inducer = Inducer(x.shape[1])
inducer(x)
