import multiprocessing as mp
from copy import deepcopy
import torch


class ESQOptimizer(object):
    
    
    def __init__(self,critic,actor,
                        sigma,params_per_step,n_workers=None):
        
        if n_workers is None : n_workers = mp.cpu_count()
        
        self.args = locals()
        self.args.pop("self")
        
        
        self.actor = actor
        self.optimizer = torch.optim.Adam(actor.model.network.parameters())
        #self.__dict__.update(locals())
        

    def run_noisy_advantage(self,states,critic,actor,
                            sigma,seed):
        
        torch.manual_seed(seed)
        n_params = actor.model.network.n_params
        eps = torch.randn(n_params) * sigma
        actor.model.network.add_noise(eps)
        #actor2 = actor.model.network.add_noise(-eps)
    
        noisy_actions = actor(states,output_tensor=True)
        noisy_q = critic(states,noisy_actions,output_tensor=True) ##  add absorbing flag
        noisy_q = torch.sum(noisy_q)
        noisy_grad = torch.autograd.grad(noisy_q,actor.model.network.parameters()) ## hotfix
        
        #return grad1+grad2
        
        return noisy_grad
        
    def run_parallel_advantage(self,states,critic,actor,
                               sigma,params_per_step,n_workers):
        
        args = [(states,critic,actor,sigma,seed) for seed in range(params_per_step)]
        
        ctx = mp.get_context('spawn')
        
        # Step 1: Init multiprocessing.Pool()

        with ctx.Pool(n_workers) as pool:
        
            # Step 2:  Run processes (we might need to use mapreduce to avoid big memory usage)
            grads = pool.starmap(self.run_noisy_advantage,args) ## list of [(reward*eps)]

            # Step 3: Wait for workers to run then close pool
            pool.close()
            pool.join()
            
        
        return grads
    
    def compute_grads(self,states):
        
        grads = self.run_parallel_advantage(**self.args,states=states)  
        
        # Step 4: Aggregate gradients
        
        grad_stack = []
        
        for i in range(len(grads[0])):
            
            grad_stack.append(torch.stack([grad[i] for grad in grads]))
            
        grads = tuple(torch.sum(stack,dim=0) for stack in grad_stack)
        
        return grads

    def step(self,states):
        
        self.optimizer.zero_grad()  
        actor_grad = self.compute_grads(states)      
        self.actor.model.network.grad = actor_grad ## optimizer usually minimizes (add -)
        self.optimizer.step()
      