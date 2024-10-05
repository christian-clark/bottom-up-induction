import numpy as np, sys

nmax = int(sys.argv[1])

# row is sentence length
# row[0] is number of trees with x0 category
# row[1] is number of trees with x1 category
# row[2] is number of trees with x2 category
c = np.zeros((nmax, 3))
c[0, 2] = 1

for n in range(1, nmax):
    x0_n = 0
    for k in range(1, n-1):
        x0_n += c[k, 0]*c[n-k, 0] # TODO continue here 

print(c)