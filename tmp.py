

import torch 
import gpytorch 
import logging
import logging.config

from src.helpers import setup_experiment
from src.trainer import Trainer
from src.optimizers.gibo import GIBOptimizer
from src.optimizers.vanilla_bo import BOptimizer
from src.config import get_configs

logging.config.fileConfig('logging.conf')
# create root logger
logger = logging.getLogger()
print("hello")


#env_name = "CartPole-v1"
env_name = "Swimmer-v4"
#env_name = "Hopper-v2"
kernel_name = "rbfstate" ## "linearstate" /"rbfstate"

env_config,likelihood_config,kernel_config,optimizer_config,trainer_config = get_configs(env_name,kernel_name)
additional_layers=[] ### can be empty or [8,7] for adding 2 layers with width 8,7 respectively

optimizer_config = {
        "n_eval":4, ## 3 for cartpole (very noisy)
        ### for GIBO
        "n_max":32, 
        "n_info_samples":16,
        "delta":0.1, ## 0.01 for cartpole
        ### hessian normalisation applies only for rbf
        "normalize_gradient":True if kernel_name == "rbf" else False,
        "standard_deviation_scaling":False,
}

likelihood_config = {
                "noise_hyperprior":gpytorch.priors.torch_priors.UniformPrior(a=0.1,b=0.101),
                "noise_constraint":gpytorch.constraints.constraints.Interval(0.1,0.101)
                }


kernel_config = {
        "use_ard":True,
        "kernel_name":kernel_name,
        #"lengthscale_hyperprior":gpytorch.priors.torch_priors.GammaPrior(5,0.9),
        "lengthscale_constraint":gpytorch.constraints.constraints.Interval(0.01,200),
        #"outputscale_hyperprior":gpytorch.priors.torch_priors.GammaPrior(2,0.4),
        "outputscale_constraint":gpytorch.constraints.constraints.GreaterThan(0.01),
        }

trainer_config = {
        "n_steps":20, 
        "report_freq":1,
        "save_best":False,
        "wandb_logger":False,
}

model,objective_env = setup_experiment(env_config,kernel_config,likelihood_config,additional_layers)


### Chose optimizer 
#optimizer = BOptimizer(**optimizer_config)
optimizer = GIBOptimizer(model,**optimizer_config)
trainer = Trainer(model,objective_env,optimizer,**trainer_config)
rslt= trainer.run()

### ADD LR SCHEDULAR  / FIX DISCRETIZATION ===> ENJOY WEEKEND :DDD