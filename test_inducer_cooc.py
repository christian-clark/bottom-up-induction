import torch
from inducer_cooc import Inducer


MAX_EPOCH = 10000
EPOCH_PRINT_FREQ = 2000


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

# word 0: chickens
# word 1: eat
# word 2: people
# word 3: seeds

x = torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

#x = torch.Tensor([
#    [0.1, 0.02, -0.03],
#    [-0.02, 0.2, 0.01],
#    [-0.01, 0.03, 0.3]
#])

inducer = Inducer(x.shape[1], operation_model_type="hacky")
# default lr: 0.001
optimizer = torch.optim.Adam(inducer.parameters())
epoch = 0
loss, combined_probs = inducer(x, print_trees=True)
combined_probs_tracking = list()
loss_tracking = list()
torch.set_printoptions(linewidth=200, precision=2)
#while epoch < 10000:
#while epoch < 1000:
while epoch < MAX_EPOCH:
    if epoch % EPOCH_PRINT_FREQ == 0:
        print("\n==== Epoch {} ====".format(epoch))
        print("loss:", loss)
        print("tree probabilities:", combined_probs)
#        print("ordering_model weight:")
#        print(inducer.ordering_model.weight)
#        print("operation_model weight:")
#        print(inducer.operation_model.weight)
        combined_probs_tracking.append(combined_probs)
        loss_tracking.append(loss)
    optimizer.zero_grad()
    if epoch % EPOCH_PRINT_FREQ == 0: v = True
    else: v = False
    loss, combined_probs = inducer(x, verbose=v)
    loss.backward()
    optimizer.step()
    epoch += 1

combined_probs_tracking = torch.stack(combined_probs_tracking, dim=0)
print("\n\n==== Tree probabilities ====")
print(combined_probs_tracking)

loss_tracking = torch.stack(loss_tracking, dim=0)
print("\n==== Loss ====")
print(loss_tracking)
