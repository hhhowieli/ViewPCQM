import torch

def expectation(distribution, dim):
    
    b = distribution.size(0)
    s = distribution.size(1)

    res = torch.zeros((b, 1))

    for i in range(s):
        res = res + (i+1)*(distribution[:, i])


def mos2distribution(mos):
    pass