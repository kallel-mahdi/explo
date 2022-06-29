import multiprocessing as mp
from copy import deepcopy

import torch


class ESOptimizer(object):
    
    
    def __init__(self,env,policy_params,
                 params_per_step,episodes_per_param,n_workers,
                 sigma):
        
        self.__dict__.update(locals())
        
        self.optimizer = torch.optim.Adam([self.policy_params])
        

    def run_noisy_actor(self,env,policy_params,
                        seed,sigma,episodes_per_param):
        
        """ Generate and run a pair of symmetric noisy parameters

        Returns:
            weighted_noise: J(theta+eps)*eps
        """
        
        ## Generate noisy parameters
        ## Noise is different for each seed
        torch.manual_seed(seed)
        eps = torch.randn_like(policy_params) * sigma
        
        params1 = policy_params + eps
        params2 = policy_params - eps ## symmetric noise

        tmp_env = deepcopy(env)
        reward1,_,transitions1 = tmp_env.run_many(params1,episodes_per_param)
        reward2,_,transitions2 = tmp_env.run_many(params2,episodes_per_param)
        
        weighted_noise = (1/sigma)*(reward1*eps - reward2*eps)*0.5 ## 
        transitions = transitions1 + transitions2
        
        return weighted_noise,transitions
    



    def run_parallel_actors(self,env,policy_params,sigma,
                    params_per_step,episodes_per_param,n_workers=None):

        args = [(env,policy_params,i,1,episodes_per_param) for i in range(params_per_step)]
        
        # Step 1: Init multiprocessing.Pool()

        if n_workers is None : n_workers = mp.cpu_count()
        pool = mp.Pool(n_workers)
        
        # Step 2:  Run processes (we might need to use mapreduce to avoid big memory usage)
        weighted_noises = pool.starmap(self.run_noisy_actor,args) ## list of [(reward*eps)]

        # Step 3: Wait for workers to run then close pool
        pool.close()
        pool.join()

        return weighted_noises,transitions
    
    def compute_gradient(self,weighted_noises):
        
        gradient  = torch.vstack(weighted_noises).mean(dim=0)
    
    
    def run(self):
        
        
        local_reward,local_states,local_transitions = self.env.run_many(self.policy_params,self.episodes_per_param)
        
        weighted_noises,transitions = self.run_parallel_actors(self.env,self.policy_params,self.sigma,
                    self.params_per_step,self.episodes_per_param)
        
        nes_policy_grad = self.compute_gradient(weighted_noises)
        
        transitions = transitions + local_transitions

        return nes_policy_grad,local_states,transitions

    
    def step(self):
        
        self.optimizer.zero_grad()        
        self.policy_params.grad = -policy_grad ## optimizer usually minimizes
        self.optimizer.step()
        
        
    

if __name__ == '__main__':
    

    #%cd /home/q123/Desktop/explo/

    import torch
    from src.config import get_configs
    from src.helpers import setup_experiment
    from src.optimizers.es_pytorch import ESOptimizer

    env_name = "Swimmer-v4"
    kernel_name = "rbf"

    env_config,likelihood_config,kernel_config,optimizer_config,trainer_config = get_configs(env_name,kernel_name)
    _,env = setup_experiment(env_config,kernel_config,likelihood_config,additional_layers=[])

    optimizer = ESOptimizer(env,torch.zeros(env.mlp.len_params),sigma=1e-2,
                    params_per_step=50,episodes_per_param=1,n_workers=8)


    for i in range(100):
        
        optimizer.step()
        
        if i % 3 == 0:
            avg_reward,_,_ = env.run_many(optimizer.policy_params,5)
            print(f'avg_rewarad {avg_reward} ')
            print(f'policy_params : {optimizer.policy_params}')
