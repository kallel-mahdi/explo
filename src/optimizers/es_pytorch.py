import multiprocessing as mp
from copy import deepcopy

import torch


class ESOptimizer(object):
    
    
    def __init__(self,env,policy_params,
                 params_per_step,episodes_per_param,n_workers,
                 sigma):
        
        self.__dict__.update(locals())
        
        self.optimizer = torch.optim.Adam([self.policy_params])
        

    def run_noisy_params(self,env,policy_params,
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
        rewards1,states1 = tmp_env.run_many(params1,episodes_per_param)
        rewards2,states2 = tmp_env.run_many(params2,episodes_per_param)
        
        weighted_noise = (1/sigma)*(rewards1*eps - rewards2*eps)*0.5 ## 
        
        return weighted_noise



    def compute_gradient(self,env,policy_params,sigma,
                    params_per_step,episodes_per_param,n_workers=None):

        args = [(env,policy_params,i,1,episodes_per_param) for i in range(params_per_step)]
        
        # Step 1: Init multiprocessing.Pool()

        if n_workers is None : n_workers = mp.cpu_count()
        pool = mp.Pool(n_workers)

        # Step 2:  Run processes (we might need to use mapreduce to avoid big memory usage)
        weighted_noises = pool.starmap(self.run_noisy_params,args) ## list of [(reward*eps)]

        # Step 3: Wait for workers to run then close pool
        pool.close()
        pool.join()

        gradient  = torch.vstack(weighted_noises).mean(dim=0)
        
        return gradient
    
    
    def step(self):
        
        
        policy_grad = self.compute_gradient(self.env,self.policy_params,self.sigma,
                    self.params_per_step,self.episodes_per_param,)
    
        self.optimizer.zero_grad()        
        self.policy_params.grad = -policy_grad ## optimizer usually minimizes
        self.optimizer.step()
        
        return policy_grad
        
        
    

if __name__ == '__main__':
    
 
    from src.optimizers.es_pytorch import ESOptimizer
    from src.helpers import setup_experiment
    from src.config import get_configs
    import torch

    env_name = "Swimmer-v4"
    kernel_name = "rbf"

    env_config,likelihood_config,kernel_config,optimizer_config,trainer_config = get_configs(env_name,kernel_name)
    _,env = setup_experiment(env_config,kernel_config,likelihood_config,additional_layers=[])

    optimizer = ESOptimizer(env,torch.zeros(18),sigma=1e-2,
                    params_per_step=50,episodes_per_param=1,n_workers=8)


    for i in range(100):
        
        optimizer.step()
        
        if i % 3 == 0:
            avg_reward,_ = env.run_many(optimizer.policy_params,5)
            print(f'avg_rewarad {avg_reward} ')
            print(f'policy_params : {optimizer.policy_params}')
