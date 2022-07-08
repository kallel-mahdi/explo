#import multiprocessing as mp
import torch.multiprocessing as pmp
from copy import deepcopy
import torch
import logging
import logging.config

class ESQOptimizer(object):
    
    
    def __init__(self,critic,actor,
                        sigma,params_per_step,n_workers=None):
        
        if n_workers is None : n_workers = mp.cpu_count()
        
        self.worker_args = locals()
        self.worker_args.pop("self")
        self.actor = actor
        self.optimizer = torch.optim.Adam(actor.parameters())
        #self.optimizer = torch.optim.SGD(actor.parameters(),lr=0.5)
        #self.__dict__.update(locals())
        

    def run_noisy_advantage(self,states,critic,actor,
                            sigma,seed):
        
        torch.manual_seed(seed)
        n_params = actor.n_params
        eps = torch.randn(n_params) * sigma
        
        
        ### define function locally or thread wont like
        def run_noisy_grad(eps):
            
            tmp_actor = deepcopy(actor)
            tmp_actor.add_noise(eps)
            noisy_actions = tmp_actor(states)
            noisy_q = critic(states,noisy_actions) ##  add absorbing flag?? (you can no longer backprop)
            noisy_q = torch.sum(noisy_q)
            noisy_grad = torch.autograd.grad(noisy_q,tmp_actor.parameters()) ## hotfix
            
            return noisy_grad
        
        grad1 = run_noisy_grad(eps)
        grad2 = run_noisy_grad(-eps)
    
        return grad1+grad2
        

    def run_parallel_advantage(self,states,critic,actor,
                               sigma,params_per_step,n_workers):
        
        
        args = [(states,critic,actor,sigma,seed) for seed in range(params_per_step)]
        
        critic.share_memory()
        actor.share_memory()
        
        ctx = pmp.get_context('spawn')
        
        with ctx.Pool(n_workers) as pool:
            
            # Step 2:  Run processes (we might need to use mapreduce to avoid big memory usage)
            grads = pool.starmap(self.run_noisy_advantage,args) ## list of [(reward*eps)]

            # Step 3: Wait for workers to run then close pool
            pool.close()
            pool.join()

        return grads
    
    def compute_grads(self,states):
        
        grads = self.run_parallel_advantage(**self.worker_args,states=states)  
        
        # Step 4 : aggregate gradients 
        
        grad_stack = []  
              
        for i in range(len(grads[0])):
            
            grad_stack.append(torch.stack([g[i] for g in grads]))
            
        ## optimizer usually minimizes (add -)
        grads = tuple(-torch.mean(stack,dim=0) for stack in grad_stack)
        
        return grads

    def step(self,states):
        
        self.optimizer.zero_grad()  
        actor_grad = self.compute_grads(states)      
        self.actor.grad = actor_grad 
        self.optimizer.step()
      