import logging
import logging.config
import random

import botorch
import gpytorch
import numpy as np
import torch
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.policy import DeterministicPolicy
from mushroom_rl.utils.replay_memory import ReplayMemory
from mushroom_rl.utils.spaces import Box, Discrete

from src.approximators.actor import MLP, ActorNetwork
from src.approximators.critic import CriticNetwork
from src.approximators.ddpg import DDPG
from src.approximators.td3 import TD3
from src.environments.gym_env import Gym
from src.environments.objective import EnvironmentObjective
from src.gp.gp import DEGP, MyGP
from src.gp.kernels import *
from src.gp.means import *
#from src.optimizers.gibo import GIBOptimizer
from src.optimizers.gibo_parallel import GIBOptimizer

logging.config.fileConfig('logging.conf')
logger = logging.getLogger("MathLog."+__name__)


def get_initial_data(mlp,objective_env,n_init):
    
    
    ### generate initial data
    #train_x = torch.rand(n_init,mlp.len_params) ## [n_trials,n_params]
    print("MLP LEEEEEN",mlp.len_params)
    train_x = torch.zeros(n_init,mlp.len_params) ## [n_trials,n_params]
    tmp = [objective_env.run(p) for p in train_x]
    train_y = torch.Tensor([d[0] for d in tmp]).reshape(-1)  ## [n_trials,1]
    train_y_disc = torch.Tensor([d[1] for d in tmp]).reshape(-1)  ## [n_trials,1]
    train_s = torch.cat( [d[2] for d in tmp])  ## [n_trials,max_len,state_dim]
    train_s = torch.flatten(train_s,start_dim=0,end_dim=1) ## [n_trials*max_len,state_dim]
    
    return (train_x,train_y,train_y_disc,train_s)

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

    print("MLPPPPPPPPP SETUP",mlp.len_params)
    
    return mlp

def setup_agent(objective_env,policy_config,
    initial_replay_size = 500,
    max_replay_size = 72000,
    n_features = 256,
    tau = .99 ##used to be 0.99
    ):
    
    # MDP
    horizon = 500
    gamma = 0.99
    gamma_eval = 1.


    # Settings
    batch_size = max_replay_size


    mdp = objective_env.env

    policy_class = DeterministicPolicy
    policy_params = dict()

    actor_input_shape = mdp.info.observation_space.shape
    actor_output_shape = mdp.info.action_space.shape
    
    actor_params = dict(network=MLP,
                        input_shape = actor_input_shape,
                        output_shape=actor_output_shape,
                        Ls=[actor_input_shape[0]]+policy_config["add_layer"]+[actor_output_shape[0]],
                        add_bias=policy_config["add_bias"])

    #actor_params = objective_env.mlp
    
    ### Eventually replace this with GIBO
    
    ### This is unused
    actor_optimizer = {
                        'class': torch.optim.Adam,
                        'params': {'lr': 1e-4}
                        }

    #####################################

    critic_input_shape = (actor_input_shape[0] + actor_output_shape[0],)


    critic_params = dict(network=CriticNetwork,
                        optimizer={'class': torch.optim.Adam,
                                    'params': {'lr': 1e-3}},
                        loss=torch.nn.functional.mse_loss,
                        n_features=n_features,
                        input_shape=critic_input_shape,
                        output_shape=(1,),
                        batch_size = 1000, ## new
                        )


    print("MDP INFOOOOOO",mdp.info.gamma)
    print("MDP INFOOOOOO",mdp.info.gamma)
    print("MDP INFOOOOOO",mdp.info.gamma)

    agent = TD3(mdp.info, policy_class,policy_params,
                actor_params, actor_optimizer, 
                critic_params,
                batch_size, initial_replay_size, max_replay_size,
                tau)
    
    return agent




def setup_kernel(kernel_config,agent,train_s):
    
    kernel_name = kernel_config.pop("kernel_name")
    
    ### If not using a statekernel: ard_num_dims = num_parameters
    ### Otherwise statekernel handles ard_num_dims dynamically
    
 
    mlp = agent._actor_approximator.model.network
    
    kernel_config["mlp"]= mlp
    
    if kernel_name == "rbf":
        
        kernel_config["ard_num_dims"]=mlp.len_params
        print(f'Using ard_num_dims = {mlp.len_params}')
        kernel = MyRBFKernel(**kernel_config)
        
    elif kernel_name == "rbfstate":
        
        kernel = RBFStateKernel(**kernel_config,train_s=train_s)

    else : raise ValueError("Unknown kernel")
    
    return kernel

def setup_mean(mean_config,agent):
    
    if mean_config["advantage"]==True:
        
        mean_module  = AdvantageMean(agent)
    
    else :
        
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
    
    train_x,train_y,train_y_disc,train_s = get_initial_data(mlp,objective_env,n_init)
    
    
    agent = setup_agent(objective_env,policy_config)
    covar_module = setup_kernel(kernel_config,agent,train_s=train_s)
    mean_module = setup_mean(mean_config,agent)
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
            **likelihood_config
        )
    
    # initialize likelihood and model
    model = DEGP(train_x=train_x,train_y=train_y,train_s=train_s,
                 mean_module = mean_module,covar_module = covar_module,
                 likelihood=likelihood)
    
    optimizer = GIBOptimizer(agent,model,**optimizer_config)

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
    
    

