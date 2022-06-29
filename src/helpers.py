import logging
import logging.config

import gpytorch
import torch

from mushroom_rl.utils.spaces import Box, Discrete

from src.environments.objective import EnvironmentObjective
from src.environments.gym_env import Gym

from src.gp import DEGP, MyGP
from src.kernels import *
from src.approximators.actor import MLP

logging.config.fileConfig('logging.conf')
logger = logging.getLogger("MathLog."+__name__)


def get_initial_data(mlp,objective_env,n_init):
    
    
    ### generate initial data
    #train_x = torch.rand(n_init,mlp.len_params) ## [n_trials,n_params]
    train_x = torch.zeros(n_init,mlp.len_params) ## [n_trials,n_params]
    tmp = [objective_env.run(p) for p in train_x]
    train_y = torch.Tensor([d[0] for d in tmp]).reshape(-1)  ## [n_trials,1]
    train_s = torch.cat( [d[1] for d in tmp])  ## [n_trials,max_len,state_dim]
    train_s = torch.flatten(train_s,start_dim=0,end_dim=1) ## [n_trials*max_len,state_dim]
    
    return (train_x,train_y,train_s)

def setup_policy(env,additional_layers):
    
    n_inputs  = env.info.observation_space.shape[0]
    action_space = env.info.action_space
    
    if type(action_space) == Discrete:
        ###output one action env will discretize
        n_actions = 1 
    elif type(action_space) == Box:
        n_actions = action_space.shape[0]
    else : raise ValueError("Unknown action space")
    
    logger.warning(f'MLP dimensions : {[n_inputs] +additional_layers + [n_actions]}')
    mlp = MLP([n_inputs]+additional_layers+[n_actions],add_bias=True)
    
    return mlp

def setup_kernel(kernel_config,mlp,train_s):
    
    kernel_name = kernel_config.pop("kernel_name")
    
    ### If not using a statekernel: ard_num_dims = num_parameters
    ### Otherwise statekernel handles ard_num_dims dynamically
    kernel_config["ard_num_dims"]=mlp.len_params
    print(f'Using ard_num_dims = {mlp.len_params}')
    
    if kernel_name == "rbf":
        
        kernel = MyRBFKernel(**kernel_config)
    
    elif kernel_name == "linearstate":
            
        kernel = LinearStateKernel(**kernel_config,mlp=mlp,train_s=train_s)
    
    elif kernel_name == "rbfstate":
        
        kernel = RBFStateKernel(**kernel_config,mlp=mlp,train_s=train_s)
        
    elif kernel_name == "maternstate":
            
        kernel = MaternStateKernel(**kernel_config,mlp=mlp,train_s=train_s)
    
    else : raise ValueError("Unknown kernel")
    
    return kernel

    
def setup_experiment(env_config,
                     kernel_config,likelihood_config,additional_layers=[]):
    
    
    ### build environment and linear policy
    n_init = env_config.pop("n_init")
    env = Gym(env_config["env_name"])
    mlp = setup_policy(env,additional_layers)
    
    ### objective env evaluates the policy episodically
    objective_env = EnvironmentObjective(
            env=env,
            mlp=mlp,
            **env_config
            )
    
    train_x,train_y,train_s = get_initial_data(mlp,objective_env,n_init)
    kernel = setup_kernel(kernel_config,mlp=mlp,train_s=train_s)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
            **likelihood_config
        )
    
    # initialize likelihood and model
    model = DEGP(train_x=train_x,train_y=train_y,train_s=train_s,
                 kernel = kernel,likelihood=likelihood,
                 mlp =mlp)
    
    return model,objective_env


