import torch
import numpy as np
from scipy.stats import chi2

def get_bound(n_dim,p):


    dx = 0.001
    x = np.arange(0, n_dim,dx)

    cdf = np.cumsum(chi2.pdf(x, df=n_dim)*dx)
    bound = max(x[cdf<=p])

    return torch.tensor(bound)


def sparsify(mean_d,var_d,p=0.01):

    ### Get knapsack capacity
    mean_d = mean_d.squeeze()
    var_d = torch.abs(var_d.squeeze())
    var_d = var_d.clamp_min(1e-9)
    n_dim = mean_d.shape[0]
    bound = get_bound(n_dim,p)

    

    ### We use a heuristic for the knapsack algorithm
    cost = (mean_d**2)/var_d 
    idx = torch.argsort(cost)
    cost = cost[idx]
    weight = 0
    last_i = 0
    
    ### While knapsack is not full add items
    for i in range(len(idx)):

        if (weight + cost[i] <= bound):
            weight+=cost[i]
            last_i = i

        else :
            break

    ### Set all the items in knapsack to 0
    tmp = mean_d.clone()
    tmp[idx[:last_i]]=0
    
    ### Report fraction of sparsified entries
    fraction = last_i / len(idx)

    print(f'n_dim {n_dim} bound {bound} fraction {fraction}')
    print(f'cost {cost}')

    return tmp.reshape(1,-1),fraction


