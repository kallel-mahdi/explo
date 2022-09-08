import re
import torch
import cvxpy as cvx
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt

def get_bound(n_dim,p):

    
    dx = 0.001
    x = np.arange(0, n_dim,dx)

    cdf = np.cumsum(chi2.pdf(x, df=n_dim)*dx)
    bound = max(x[cdf<=p])

    return torch.tensor(bound)


# def sparsify(mean_d,var_d,p=0.01):

#     ### Get knapsack capacity
#     mean_d = mean_d.squeeze()
#     var_d = torch.abs(var_d.squeeze())
#     var_d = var_d.clamp_min(1e-9)
#     n_dim = mean_d.shape[0]
#     bound = get_bound(n_dim,p)

    

#     ### We use a heuristic for the knapsack algorithm
#     cost = (mean_d**2)/var_d 
#     idx = torch.argsort(cost)
#     cost = cost[idx]
#     weight = 0
#     last_i = 0
    
#     ### While knapsack is not full add items
#     for i in range(len(idx)):

#         if (weight + cost[i] <= bound):
#             weight+=cost[i]
#             last_i = i

#         else :
#             break

#     ### Set all the items in knapsack to 0
#     tmp = mean_d.clone()
#     tmp[idx[:last_i]]=0
    
#     ### Report fraction of sparsified entries
#     fraction = last_i / len(idx)

#     print(f'n_dim {n_dim} bound {bound} fraction {fraction}')
#     print(f'cost {cost}')

#     return tmp.reshape(1,-1),fraction


def sparsify(mu,Sigma,p=0.01):


    mu = mu.squeeze().detach().cpu().numpy()
    Sigma = Sigma.squeeze().detach().cpu().numpy()

    n_dim = mu.shape[0]
    
    eigvals,P = np.linalg.eigh(Sigma)
    #assert (eigvals >= 0).all()

    
    ### Square root of Sigma_inv

    D = np.diag(1/eigvals)
    D_sqrt = np.diag(np.sqrt(1/eigvals))
    P_p = P.T@D_sqrt 
    
    ###################

    C = get_bound(n_dim,p)

    x = cvx.Variable((n_dim))
    objective = cvx.Minimize(cvx.norm(x,1))
    pb = cvx.Problem(objective,
                    [cvx.sum_squares(P_p@x -P_p@mu) <= C])

    pb.solve(verbose=False)

    rslt = x.value


    relative_diff = abs(rslt-mu)/abs(mu)
    plt.hist(relative_diff)
    plt.show()


    mu[relative_diff>0.9]=0.
    mu = torch.tensor(mu).reshape(1,-1)
    sparsity = np.sum(relative_diff<=0.9)/len(relative_diff)
    rslt = mu
   
    # rslt[np.abs(rslt)<1e-5]=0
    # sparsity = np.sum(rslt==0)/len(rslt)

    # rslt = torch.tensor(rslt).float().reshape(1,-1)

    print(sparsity)
    print(rslt)
    

    return rslt,sparsity


