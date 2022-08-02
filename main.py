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

logging.config.fileConfig('logging.conf')
# create root logger
logger = logging.getLogger()



simplefilter(action='ignore', category=DeprecationWarning)
os.environ["WANDB_API_KEY"]="28996bd59f1ba2c5a8c3f2cc23d8673c327ae230"

def run(seed):

        
        #env_name = "CartPole-v1" ## Action kernel + State_norm looks very well for cartpole
        #env_name = "Swimmer-v4" ##  State_norm stabilizes training 
        env_name = "Hopper-v2"
        #env_name = "Walker2d-v3"


        kernel_name = "rbfstate" ## "rbf"
        #kernel_name = "rbf" ## "rbf"

        conf_grad = False
        norm_grad = True
        advantage_mean = True

        run_name = kernel_name + str(seed)+"_conf_grad"*conf_grad+"_norm_grad"*norm_grad+"_advantage"*advantage_mean
        env_config,policy_config,likelihood_config,kernel_config,mean_config,optimizer_config,trainer_config = get_configs(env_name,kernel_name,
        use_ard=True,manipulate_state=True,
        conf_grad=conf_grad,norm_grad=norm_grad,advantage_mean=advantage_mean,
        wandb_logger=True,run_name=run_name)

        model,objective_env,optimizer = setup_experiment(env_config,mean_config,kernel_config,likelihood_config,policy_config,optimizer_config,
                                        seed=seed)

        trainer = Trainer(model,objective_env,optimizer,**trainer_config)
        trainer.run()

        ### ADD LR SCHEDULAR ===> ENJOY WEEKEND :DDD


if __name__ == '__main__':

        
        wandb.require("service")
        
        wandb.setup()  
        np.random.seed(42)## then 41

        n = 5           
        seeds = np.random.randint(low=0,high=2**30,size=(n,))
        seeds = [ int(i) for i in seeds]

        with Pool(processes=n) as p:
                p.map(run, seeds)

        # threads = list()
        # for index in range(n):
        #         x = threading.Thread(target=run, args=(seeds[index],))
        #         threads.append(x)
        #         x.start()
        
        # for index, thread in enumerate(threads):

        #         logging.info("Main    : before joining thread %d.", index)
        #         thread.join()
        #         logging.info("Main    : thread %d done", index)
        