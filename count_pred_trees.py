import math, numpy as np, sys

nmax = int(sys.argv[1])

# row is sentence length
# row[0] is number of trees with x0 category
# row[1] is number of trees with x1 category
# row[2] is number of trees with x2 category

# the sum of row n of c gives the number of possible pred trees with n+1
# leaves, with the leaves fixed in one particular order. Need to multiply
# by (n+1)! to get the total
c = np.zeros((nmax, 3), dtype=int)
c[0, 2] = 1

for n in range(1, nmax):
    x0_n = 0
    for k in range(0, n):
        x0_n += c[k,0]*c[n-k-1,0] \
        + 2*c[k,1]*c[n-k-1,0] \
        + c[k,1]*c[n-k-1,1] \
        + c[k,1]*c[n-k-1,2] \
        + 2*c[k,2]*c[n-k-1,0] \
        + c[k,2]*c[n-k-1,1] \
        + c[k,2]*c[n-k-1,2]
    c[n, 0] = x0_n

    x1_n = 0
    for k in range(0, n):
        x1_n += c[k,0]*c[n-k-1,1] \
        + c[k,1]*c[n-k-1,1] \
        + c[k,2]*c[n-k-1,0] \
        + 2*c[k,2]*c[n-k-1,1] \
        + c[k,2]*c[n-k-1,2]
    c[n, 1] = x1_n

    x2_n = 0
    for k in range(0, n):
        x2_n += c[k,0]*c[n-k-1,2] \
        + c[k,1]*c[n-k-1,2] \
        + c[k,2]*c[n-k-1,2]
    c[n, 2] = x2_n


totals = c.sum(axis=1)
for t in totals:
    print(t)
# this is the full total when all permutations of leaves are considered
#for n, t in enumerate(totals):
#    print(t * math.factorial(n+1))
