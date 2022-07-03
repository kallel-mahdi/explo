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
        
        self.args = locals()
        self.args.pop("self")
        
        
    
        self.actor = actor
        self.optimizer = torch.optim.Adam(actor.parameters())
        #self.__dict__.update(locals())
        

    def run_noisy_advantage(self,states,critic,actor,
                            sigma,seed,grad_buffer=None):
        
        torch.manual_seed(seed)
        n_params = actor.n_params
        eps = torch.randn(n_params) * sigma
        actor.add_noise(eps)
        #actor2 = actor.add_noise(-eps)
    
        # noisy_actions = actor(states,output_tensor=True)
        # noisy_q = critic(states,noisy_actions,output_tensor=True) ##  add absorbing flag
        noisy_actions = actor(states)
        noisy_q = critic(states,noisy_actions) ##  add absorbing flag
        noisy_q = torch.sum(noisy_q)
        noisy_grad = torch.autograd.grad(noisy_q,actor.parameters()) ## hotfix
        
        #return grad1+grad2
        
        if grad_buffer : grad_buffer.append(noisy_grad)
        
        return noisy_grad
    
    def run_parallel_advantage_p(self,states,critic,actor,
                               sigma,params_per_step,n_workers):
        
        processes = []
        self.grad_buffer = []
        #grad_buffer = None
        
        args = [(states,critic,actor,sigma,seed,self.grad_buffer) for seed in range(params_per_step)]
        
        critic.share_memory()
        actor.share_memory()
        
        ctx = pmp.get_context('spawn')
        
        for i in range(n_workers):
            
            p = ctx.Process(target=self.run_noisy_advantage, args=args[i])
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
            
        return self.grad_buffer
    
        
    def run_parallel_advantage(self,states,critic,actor,
                               sigma,params_per_step,n_workers):
        
        args = [(states,critic,actor,sigma,seed) for seed in range(params_per_step)]
       
        ctx = mp.get_context('fork')
        
        # Step 1: Init multiprocessing.Pool()

        with ctx.Pool(n_workers) as pool:
        
            # Step 2:  Run processes (we might need to use mapreduce to avoid big memory usage)
            grads = pool.starmap(self.run_noisy_advantage,args) ## list of [(reward*eps)]

            # Step 3: Wait for workers to run then close pool
            pool.close()
            pool.join()
            
        
        return grads
    
    def compute_grads(self,states,aggregate=False):
        
        print("running parallel advantage")
        grads = self.run_parallel_advantage_p(**self.args,states=states)  
        print("done running parallel advantage")
        #grads = self.run_parallel_advantage(**self.args,states=states)  
        if aggregate:
            
            grad_stack = []
            
            for i in range(len(grads[0])):
                
                grad_stack.append(torch.stack([grad[i] for grad in grads]))
                
            grads = tuple(torch.sum(stack,dim=0) for stack in grad_stack)
            
        return grads

    def step(self,states):
        
        self.optimizer.zero_grad()  
        actor_grad = self.compute_grads(states)      
        self.actor.grad = actor_grad ## optimizer usually minimizes (add -)
        self.optimizer.step()
      