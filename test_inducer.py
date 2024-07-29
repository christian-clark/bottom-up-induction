import torch
from inducer import Inducer

# word vectors:
#
#           animate eating  edible
# people    1       0       0
# eat       0       1       0
# donuts    0       0       1

#role1(func, arg):
#func[eating] AND arg[animate]

#role2(func, arg):
#func[eating] AND arg[edible]

#composition: just keep functor vector


x = torch.Tensor([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

inducer = Inducer(x.shape[1])
optimizer = torch.optim.Adam(inducer.parameters(), lr=0.01)
epoch = 0
loss, combined_probs = inducer(x, print_trees=True)
while epoch < 10000:
    if epoch % 200 == 0:
        print("==== epoch {} ====".format(epoch))
        print("loss:", loss)
        print("combined_probs:", combined_probs)
    optimizer.zero_grad()
    loss, combined_probs = inducer(x)
    loss.backward()
    optimizer.step()
    epoch += 1

