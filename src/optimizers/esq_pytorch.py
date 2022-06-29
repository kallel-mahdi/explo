import multiprocessing as mp
from copy import deepcopy
import torch


class ESQOptimizer(object):
    
    
    def __init__(self,states,critic,actor_mlp,actor_params,
                        sigma,params_per_step,n_workers=None):
        
        if n_workers is None : n_workers = mp.cpu_count()
        
        self.args = locals()
        self.args.pop("self")
        
        optimizer = torch.optim.Adam([actor_params])
        
         
    def advantage_gradient(self,states,critic,actor_mlp,actor_params):
        
        noisy_actions = actor_mlp(actor_params,states).squeeze().T
        noisy_q = critic(states,noisy_actions,output_tensor=True) ##  add absorbing flag
        noisy_q = torch.sum(noisy_q)
        noisy_grad = torch.autograd.grad(noisy_q,actor_params)
        
        return noisy_grad

    def run_noisy_advantage(self,states,critic,actor_mlp,actor_params,
                            sigma,seed):
        
        
        
        torch.manual_seed(seed)
        eps = torch.randn_like(actor_params) * sigma
        
        actor_params1 = actor_params + eps
        actor_params2 = actor_params - eps
        
        #grad1 = self.advantage_gradient(states,critic,actor_mlp,actor_params1)
        #grad2 = self.advantage_gradient(states,critic,actor_mlp,actor_params2)
        noisy_actions = actor_mlp(actor_params1,states).squeeze().T
        noisy_q = critic(states,noisy_actions,output_tensor=True) ##  add absorbing flag
        noisy_q = torch.sum(noisy_q)
        noisy_grad = torch.autograd.grad(noisy_q,actor_params)
        
        #return grad1+grad2
        
        return noisy_grad
        
    def run_parallel_advantage(self,states,critic,actor_mlp,actor_params,
                               sigma,params_per_step,n_workers):
        
        
    
        args = [(states,critic,actor_mlp,actor_params,sigma,seed) for seed in range(params_per_step)]
        
        ctx = mp.get_context('fork')
        
        # Step 1: Init multiprocessing.Pool()

        with ctx.Pool(n_workers) as pool:
        
            # Step 2:  Run processes (we might need to use mapreduce to avoid big memory usage)
            grads = pool.starmap(self.run_noisy_advantage,args) ## list of [(reward*eps)]

            # Step 3: Wait for workers to run then close pool
            pool.close()
            pool.join()

        return grads
    
    def run(self):
        
        grads = self.run_parallel_advantage(**self.args)
        
        return grads
        