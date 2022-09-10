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
#     var_d = torch.diag(var_d.squeeze())
#     var_d = torch.abs(var_d)
#     #var_d = var_d.clamp_min(1e-9)
#     assert all(var_d>=0)
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








def sparsify(mu,Sigma,p=0.05):


    mu = mu.squeeze().detach().cpu().numpy()
    Sigma = Sigma.squeeze().detach().cpu().numpy()
    n_dim = mu.shape[0]
    
    ### Method 1 for root
    # eigvals,P = np.linalg.eigh(Sigma)
    # eigvals[eigvals<=0] = 1e-4
    # D_inv_sqrt = np.diag(np.sqrt(1/eigvals))
    # P_p = (P @ D_inv_sqrt).T
    
    ### Method 2 for root

    
    m = np.min(np.abs(Sigma))
    Sigma_b = Sigma / m
    Sigma_inv = np.linalg.inv(Sigma_b)/ m
    Sigma_inv2 = np.linalg.inv(Sigma)

    P_b = np.linalg.cholesky(Sigma_b)
    P_b_inv = np.linalg.inv(P_b)
    P_p = P_b_inv / np.sqrt(m)
    
    ###########################

        
    p = p/n_dim### Temporary fix for high dimensions
    C = get_bound(n_dim,p)

    x = cvx.Variable((n_dim))
    objective = cvx.Minimize(cvx.norm(x,1))
    pb = cvx.Problem(objective,
                    [cvx.sum_squares(P_p@x -P_p@mu) <= C])

    pb.solve(verbose=False)

    rslt = x.value
    relative_diff = abs(rslt-mu)/abs(mu)


    #### METHOD 1
    mu[relative_diff>0.95]=0.
    mu = torch.tensor(mu).reshape(1,-1)    
    sparsity = np.sum(relative_diff<=0.95)/len(relative_diff)
    #########################

    ### METHOD 2
    # rslt[relative_diff>0.95]=0.
    # rslt = torch.tensor(rslt).float().reshape(1,-1)
    # sparsity = np.sum(relative_diff<=0.95)/len(relative_diff)
    # mu = rslt
    #########################

    # print(sparsity)
    # plt.hist(relative_diff)
    # plt.show()

    return mu,sparsity

