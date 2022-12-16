import logging
import logging.config
import random

import botorch
import gpytorch
import numpy as np
import torch
from mushroom_rl.utils.spaces import Box, Discrete

from src.environments.gym_env import Gym
from src.environments.objective import EnvironmentObjective
from src.gp.gp import DEGP, MyGP
from src.gp.kernels import *
from src.gp.means import *
from src.actors.mlp import *
from src.optimizers.gibo_parallel import GIBOptimizer

logging.config.fileConfig('logging.conf')
logger = logging.getLogger("MathLog."+__name__)


def get_initial_data(mlp,objective_env,n_init):
    
    print("MLP NUM PARAMS",mlp.len_params)
    
    ### In GIBO and ARS they always start from 0 initial data
    #train_x = torch.rand(n_init,mlp.len_params) ## [n_trials,n_params]
    train_x = torch.zeros(1,mlp.len_params) ## [n_trials,n_params]
    
    
    tmp = [objective_env.run(p) for p in train_x]
    train_y = torch.Tensor([d[0] for d in tmp]).reshape(-1)  ## [n_trials,1]
    train_s = torch.cat( [d[1] for d in tmp])  ## [n_trials,max_len,state_dim]
    train_s = torch.flatten(train_s,start_dim=0,end_dim=1) ## [n_trials*max_len,state_dim]
    
    return (train_x,train_y,train_s)

def setup_policy(env,policy_config):
    
    n_inputs  = env.info.observation_space.shape[0]
    action_space = env.info.action_space
    
    if type(action_space) == Discrete:
        ###output one action env will discretize
        n_actions = 1 
    elif type(action_space) == Box:
        n_actions = action_space.shape[0]
    else : raise ValueError("Unknown action space")
    
    mlp = MLP(input_shape=None,output_shape=None,Ls=[n_inputs]+policy_config["add_layer"]+[n_actions],add_bias=policy_config["add_bias"])
    
    logger.warning(f'MLP dimensions : {[n_inputs] +policy_config["add_layer"] + [n_actions]}')

    
    return mlp


def setup_kernel(kernel_config,mlp,train_s):
    
    kernel_name = kernel_config.pop("kernel_name")
    kernel_config["mlp"]= mlp
    
    ### If not using a statekernel: ard_num_dims = num_parameters
    if kernel_name == "rbf":
        
        kernel_config["ard_num_dims"]=mlp.len_params
        kernel = MyRBFKernel(**kernel_config)
        
    ### Otherwise statekernel handles ard_num_dims dynamically
    elif kernel_name == "rbfstate":
        
        kernel = RBFStateKernel(**kernel_config,train_s=train_s)

    else : raise ValueError("Unknown kernel")
    
    return kernel

def setup_mean(mean_config):
    
        mean_module = MyConstantMean()
    
        return mean_module
    
def setup_experiment(env_config,
                     mean_config,kernel_config,
                     likelihood_config,policy_config,
                    optimizer_config,seed=None):
    
    
    ### build environment and linear policy
    n_init = 1  
    env = Gym(env_config["env_name"])
    mlp = setup_policy(env,policy_config)
    
    ### objective env evaluates the policy episodically
    objective_env = EnvironmentObjective(
            env=env,
            mlp=mlp,
            **env_config
            )
    
    train_x,train_y,train_s = get_initial_data(mlp,objective_env,n_init)
    covar_module = setup_kernel(kernel_config,mlp,train_s=train_s)
    mean_module = setup_mean(mean_config)
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
            **likelihood_config)
    
    # initialize likelihood and model
    model = DEGP(train_x=train_x,train_y=train_y,train_s=train_s,
                 mean_module = mean_module,covar_module = covar_module,
                 likelihood=likelihood)
    
    optimizer = GIBOptimizer(train_x,model,**optimizer_config)

    if seed is not None :
        
        fix_seed(objective_env,seed)
    
    return model,objective_env,optimizer


def fix_seed(objective_env,seed):
    
    print("fixing seed to ",seed)
    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    botorch.utils.sampling.manual_seed(seed=seed)
    objective_env.env.seed(seed)
    
    

