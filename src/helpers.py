import logging
import logging.config

import gpytorch
import gym
import torch
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from src.environment import EnvironmentObjective
from src.gp import DEGP, MyGP
from src.kernels import *
from src.policy import MLP

logging.config.fileConfig('logging.conf')
logger = logging.getLogger("MathLog."+__name__)



def get_initial_data(mlp,objective_env,n_init):
    
    
    ### generate initial data
    #train_x = torch.rand(n_init,mlp.len_params) ## [n_trials,n_params]
    train_x = torch.zeros(n_init,mlp.len_params) ## [n_trials,n_params]
    tmp = [objective_env.run(p) for p in train_x]
    train_y = torch.Tensor([d[0] for d in tmp]).reshape(-1)  ## [n_trials,1]
    train_s = torch.stack( [d[1] for d in tmp])  ## [n_trials,max_len,state_dim]
    train_s = torch.flatten(train_s,start_dim=0,end_dim=1) ## [n_trials*max_len,state_dim]
    
    return (train_x,train_y,train_s)

def setup_policy(env):
    
    n_inputs  = env.observation_space.shape[0]      
    action_space = env.action_space
    
    if type(action_space) == Discrete:
        ###output one action env will discretize
        n_actions = 1 
    elif type(action_space) == Box:
        n_actions = action_space.shape[0]
    else : raise ValueError("Unknown action space")
    
    logger.warning(f'MLP dimensions : {[n_inputs,n_actions]}')
    mlp = MLP([n_inputs,n_actions],add_bias=True)
    
    return mlp
    
def setup_experiment(env_config,
                     kernel_config,likelihood_config):
    
    ### build environment and linear policy
    n_init = env_config.pop("n_init")
    env = gym.make(env_config["env_name"])
    mlp = setup_policy(env)
    
    ### objective env evaluates the policy episodically
    objective_env = EnvironmentObjective(
            env=env,
            mlp=mlp,
            **env_config
            )
    
    
    train_x,train_y,train_s = get_initial_data(mlp,objective_env,n_init)
    
    # initialize likelihood and model
    model = DEGP(train_x=train_x,train_y=train_y,train_s=train_s,
                 kernel_config=kernel_config,likelihood_config=likelihood_config,
                 mlp =mlp)
    
    return model,objective_env


