#%cd /home/mkallel/explo/

import logging
import logging.config
import os
from multiprocessing import Pool
from warnings import simplefilter

import gpytorch
import numpy as np
import torch
import threading
import wandb
from src.config import get_configs
from src.helpers import setup_experiment
from src.trainer import Trainer
import itertools


logging.config.fileConfig('logging.conf')
# create root logger
logger = logging.getLogger()

simplefilter(action='ignore', category=DeprecationWarning)
os.environ["WANDB_API_KEY"]="28996bd59f1ba2c5a8c3f2cc23d8673c327ae230"
#os.environ["WANDB_API_KEY"]="baf5048ab2de89a0b2cea751b1b28ff2bfa4c782"

def run(seed,
        env_name,
        kernel_name,
        norm_grad,
        advantage_mean,
        conf_grad,
        manipulate_state,
        
        adaptive_lr,
        lr ):

        #env_name = "CartPole-v1" ## Action kernel + State_norm looks very well for cartpole
        #env_name = "Swimmer-v4" ##  State_norm stabilizes training 
        #env_name = "Hopper-v2"
        #env_name = "HalfCheetah-v2"        
        #env_name = "Walker2d-v3"

        #kernel_name = "rbfstate" ## "rbf"
        #kernel_name = "rbf" ## "rbf"

        project_name = env_name+("RBFACTION")
        run_name =  kernel_name+"_lr="+str(lr) +"_"+ str(1 *norm_grad)+ str(1 *advantage_mean)+ str(1 *conf_grad) +str(1 *manipulate_state)+ str(1 *adaptive_lr) +"_"+ str(seed)
        env_config,policy_config,likelihood_config,kernel_config,mean_config,optimizer_config,trainer_config = get_configs(env_name,kernel_name,
        use_ard=True,manipulate_state=manipulate_state,
        conf_grad=conf_grad,norm_grad=norm_grad,advantage_mean=advantage_mean,adaptive_lr=adaptive_lr,lr=lr,
        wandb_logger=True,project_name=project_name,run_name=run_name)

        model,objective_env,optimizer = setup_experiment(env_config,mean_config,kernel_config,likelihood_config,policy_config,optimizer_config,
                                        seed=seed)

        trainer = Trainer(model,objective_env,optimizer,**trainer_config)
        trainer.run()



if __name__ == '__main__':

        
        wandb.require("service")
        wandb.setup()  

        env_name = ["Swimmer-v4","Hopper-v2"]
        #env_name = ["CartPole-v1"]
        kernel_name = ["rbfstate"]
        
        norm_grad = [False,True]
        advantage_mean = [False]
        conf_grad = [False] ##run this for rbf
        manipulate_state = [False]
        adaptive_lr = [False]
        
        learning_rate = 0.2 if kernel_name == ["rbfstate"] else 0.5
        lr = [learning_rate]
        
        n= 10
        np.random.seed(42)
        seeds = np.random.randint(low=0,high=2**30,size=(n,))

        for config in itertools.product(*[env_name,kernel_name,norm_grad,advantage_mean,conf_grad,manipulate_state,adaptive_lr,lr]):

                seeds = [ int(i) for i in seeds]

                with Pool(processes=n) as p:
                        args = [(seed,*config) for seed in seeds]
                        p.starmap(run, args)

                #run(*(0,*config))

        