#%cd /home/mkallel/explo/

import logging
import logging.config
import os
from multiprocessing import Pool
from warnings import simplefilter

import gpytorch
import numpy as np
import torch

import wandb
from src.config import get_configs
from src.helpers import setup_experiment
from src.trainer import Trainer

logging.config.fileConfig('logging.conf')
# create root logger
logger = logging.getLogger()



simplefilter(action='ignore', category=DeprecationWarning)
os.environ["WANDB_API_KEY"]="28996bd59f1ba2c5a8c3f2cc23d8673c327ae230"

def f(seed):

        torch.set_num_threads(1)
                
        #env_name = "CartPole-v1" ## Action kernel + State_norm looks very well for cartpole
        env_name = "Swimmer-v4" ##  State_norm stabilizes training 
        #env_name = "Hopper-v2"
        #env_name = "Walker2d-v3"
        kernel_name = "rbfstate" ## "linearstate" /"rbfstate"

        env_config,likelihood_config,kernel_config,mean_config,optimizer_config,trainer_config = get_configs(env_name,kernel_name)
        env_config["manipulate_state"] = True

        optimizer_config = {
                "n_eval":2, ## 3 for cartpole (very noisy)
                ### for GIBO
                "n_max":32, 
                "n_info_samples":16,
                "delta":0.1, ## 0.01 for cartpole
                ### hessian normalisation applies only for rbf
                "normalize_gradient": True,
                "standard_deviation_scaling":False,
        }


        likelihood_config = {
                        "noise_hyperprior":gpytorch.priors.torch_priors.UniformPrior(a=0.01,b=0.05),
                        "noise_constraint":gpytorch.constraints.constraints.Interval(0.01,0.05)
                        }


        kernel_config = {
                "use_ard":True,
                "kernel_name":kernel_name,
                #"lengthscale_hyperprior":gpytorch.priors.torch_priors.GammaPrior(2,0.2),
                #"lengthscale_hyperprior":gpytorch.priors.torch_priors.NormalPrior(1,0,),
                "lengthscale_constraint":gpytorch.constraints.constraints.Interval(0.1,10),
                #"outputscale_hyperprior":gpytorch.priors.torch_priors.GammaPrior(2,0.4),
                "outputscale_constraint":gpytorch.constraints.constraints.GreaterThan(0.01),
                }

        mean_config = {
                        "advantage":False,
                }


        policy_config = {
                "add_layer":[],### can be empty or [8,7] for adding 2 layers with width 8,7  neurons respectively
                "add_bias":False,
        }

        trainer_config = {
                "n_steps":1000, 
                "report_freq":100,
                "save_best":False,
                "wandb_logger":True,
                "project_name":env_name,
                "run_name" : kernel_name+"_normalize_gradient_"*optimizer_config["normalize_gradient"]+str(seed),
                "wandb_config": {**env_config,**optimizer_config,**likelihood_config,**kernel_config,**policy_config}
        }


        model,objective_env,optimizer = setup_experiment(env_config,mean_config,kernel_config,likelihood_config,policy_config,optimizer_config,
                                        seed=seed)

        trainer = Trainer(model,objective_env,optimizer,**trainer_config)
        rslt= trainer.run()

        ### ADD LR SCHEDULAR  / FIX DISCRETIZATION ===> ENJOY WEEKEND :DDD


if __name__ == '__main__':

        wandb.require("service")

        torch.set_num_threads(1)
        np.random.seed(42)

        n = 25
        seeds = np.random.randint(low=0,high=2**30,size=(n,))
        seeds = [ int(i) for i in seeds]

        with Pool(processes=n) as p:
                p.map(f, seeds)

        