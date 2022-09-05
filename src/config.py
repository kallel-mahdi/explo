
import gpytorch

def get_env_configs(env_name,manipulate_state):
    
        if env_name == "CartPole-v1":

                env_config = {
                        
                        "reward_scale":500,
                        "reward_shift":0,
                        "env_name":"CartPole-v1",
                }

                env_appx_config = {
                        
                        "n_max":20, ## number of samples used to fit gp
                        "n_info": 8, ## number of samples collected to computed local gradient
                        "n_steps":500,
                        "n_eval":1
                }
                
        elif env_name == "Swimmer-v4":

                env_config = {
                        
                        "reward_scale":350,
                        "reward_shift":0,
                        "env_name":"Swimmer-v4",
                        
                }

                env_appx_config = {
                        
                        "n_max":32,
                        "n_info": 16,
                        "n_steps":600,
                        "n_eval":1
                }
        

        elif env_name == "Hopper-v2":

                env_config = {
                        
                        "reward_scale":2000, 
                        "reward_shift":-1,
                        "env_name":"Hopper-v2",
                        
                }
        
    

                env_appx_config = {
                        
                        "n_max":48,      
                        "n_info": 8,

                        # "n_max":40,      
                        # "n_info": 40,
                        "n_steps":1000,
                        "n_eval":1
                        
                }
        
        elif env_name == "HalfCheetah-v2":

                env_config = {

                        
                        "reward_scale":1000,
                        "reward_shift":0,
                        "env_name":"HalfCheetah-v2",
                        
                        
                }

                env_appx_config = {
                        
                        "n_max":48,
                        "n_info": 8,
                        "n_steps":2000,
                        "n_eval":2
                }

        
        elif env_name == "Walker2d-v3":
    
                env_config = {
                        
                        "reward_scale":1000, 
                        "reward_shift":-1,
                        "env_name":"Walker2d-v3",
                }
        

                env_appx_config = {
                        
                        "n_max":64,      
                        "n_info": 32,
                        "n_steps":4000,
                        "n_eval":2,
                        
                }
        
        elif env_name == "Ant-v4":
    
                env_config = {
                        
                        "reward_scale":1000, 
                        "reward_shift":-1,
                        "env_name":"Ant-v4",
                }
        

                env_appx_config = {
                        
                        "n_max":80,      
                        "n_info": 40,
                        "n_steps":4000,
                        "n_eval" : 2,
                        
                }


        else : raiseValueError("Unknown environment")

        env_config["manipulate_state"] = manipulate_state
    
        return env_config,env_appx_config



def get_configs(env_name,kernel_name,
        use_ard,manipulate_state,conf_grad,norm_grad,advantage_mean,adaptive_lr,lr,
        wandb_logger=False,project_name=None,run_name=None):


        
        env_config,env_appx_config = get_env_configs(env_name,manipulate_state)

        
        policy_config = {
                "add_layer":[],### can be empty or [8,7] for adding 2 layers with width 8,7  neurons respectively
                "add_bias":True, ### newwww
        }

        if env_name == "CartPole-v1": ## cartpole is a very noisy task
        
                # if "state" in kernel_name:
                
                #         likelihood_config = {
                #                 "noise_hyperprior":gpytorch.priors.torch_priors.UniformPrior(a=0.01,b=0.101),
                #                 "noise_constraint":gpytorch.constraints.constraints.Interval(0.01,0.101)
                #                 }
                
                # else :

                        likelihood_config = {
                                "noise_hyperprior":gpytorch.priors.torch_priors.UniformPrior(a=0.5,b=0.505),
                                "noise_constraint":gpytorch.constraints.constraints.Interval(0.5,0.505)
                                }
             
        else : 
                likelihood_config = {
                "noise_hyperprior":gpytorch.priors.torch_priors.UniformPrior(a=0.01,b=0.05),
                "noise_constraint":gpytorch.constraints.constraints.Interval(0.01,0.05)
                
                }

        kernel_config = {
                "use_ard":use_ard,
                "kernel_name":kernel_name,
                
                }

        if "state" in kernel_name:

                kernel_config.update({
                        #"lengthscale_hyperprior":gpytorch.priors.torch_priors.GammaPrior(1.2,0.2), ## 1.5,0.5
                        "lengthscale_hyperprior":gpytorch.priors.torch_priors.UniformPrior(a=0.01,b=2),
                        "lengthscale_constraint":gpytorch.constraints.constraints.Interval(0.01,2), ## constraints are loose to avoid crash
                        "outputscale_hyperprior":gpytorch.priors.torch_priors.UniformPrior(a=0.01,b=2),
                        "outputscale_constraint":gpytorch.constraints.constraints.Interval(0.01,2), #0.1 seems to be working fine :o                        
                })
                
        else :

                kernel_config.update({
                        "lengthscale_hyperprior":gpytorch.priors.torch_priors.UniformPrior(a=0.01,b=0.3),
                        "lengthscale_constraint":gpytorch.constraints.constraints.Interval(0.01,0.3), ## constraints are loose to avoid crash
                        "outputscale_hyperprior":gpytorch.priors.torch_priors.NormalPrior(loc=2.0,scale=1.0),
                        "outputscale_constraint":gpytorch.constraints.constraints.GreaterThan(0.01),
                })
        
        mean_config = {
                "advantage":advantage_mean,
        }



        optimizer_config = {
                "n_eval":env_appx_config["n_eval"], 
                ### for GIBO
                "n_max":env_appx_config["n_max"], 
                "n_info_samples":env_appx_config["n_info"],
                #"delta":0.2 if "Cheetah" in env_name else 0.1, ## default is 0.1
                "delta":0.1,
                "learning_rate": lr, ## default is 0.5, we used 0.1 for ablation
                "confidence_gradient":conf_grad,
                "normalize_gradient":norm_grad,
                "adaptive_lr":adaptive_lr,
                
        }


        trainer_config = {
                "n_steps":env_appx_config["n_steps"] ,
                "report_freq":100,
                "save_best":False,
                "wandb_logger":wandb_logger,
                "project_name":project_name,
                "run_name" : run_name,
                "wandb_config": {**env_config,**optimizer_config,**likelihood_config,**kernel_config,**policy_config}
        }

        
        return env_config,policy_config,likelihood_config,kernel_config,mean_config,optimizer_config,trainer_config