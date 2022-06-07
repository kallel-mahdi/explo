
import gpytorch

def get_env_configs(env_name):
    
        if env_name == "CartPole-v1":

                env_config = {
                        "n_init" : 1,
                        "reward_scale":500,
                        "reward_shift":0,
                        "env_name":"CartPole-v1",
                }

                env_appx_config = {
                        "ard_num_dims":5,## for kernel
                        "n_max":20,## for giboptimizer
                        "n_info": 8
                }


        elif env_name == "Swimmer-v2":

                env_config = {
                        "n_init" : 1,
                        "reward_scale":350,
                        "reward_shift":0,
                        "env_name":"Swimmer-v2",
                        
                }

                env_appx_config = {
                        "ard_num_dims":18,
                        "n_max":32,
                        "n_info": 16,
                }

        elif env_name == "Hopper-v2":

                env_config = {
                        "n_init" : 1,
                        "reward_scale":1000,
                        "reward_shift":1,
                        "env_name":"Hopper-v2",
                }

                env_appx_config = {
                        
                        "ard_num_dims":36,
                        "n_max":48,      
                        "n_info": 8,
                        
                }


        else : raiseValueError("Unknown environment")
    
        return env_config,env_appx_config



def get_configs(env_name,kernel_name):


        
        env_config,env_appx_config = get_env_configs(env_name)

        if env_name == "CartPole-v1": ## cartpole is a very noisy task
                
                likelihood_config = {
                        "noise_hyperprior":gpytorch.priors.torch_priors.UniformPrior(a=0.2,b=0.4),
                        "noise_constraint":gpytorch.constraints.constraints.Interval(0.2,0.4)
                        }
        
        else : 
                    likelihood_config = {
                        "noise_hyperprior":gpytorch.priors.torch_priors.UniformPrior(a=0.01,b=0.02),
                        "noise_constraint":gpytorch.constraints.constraints.Interval(0.01,0.02)
                        }


        kernel_config = {
                "use_ard":True,
                "kernel_name":kernel_name,
                ## in case of gridkernel ard_num_dims is number of state samples
                "ard_num_dims": 1000 if kernel_name =="grid" else env_appx_config["ard_num_dims"],
                "lengthscale_hyperprior":gpytorch.priors.torch_priors.GammaPrior(3.0,6.0),
                "lengthscale_constraint":gpytorch.constraints.constraints.GreaterThan(0.001),
                "outputscale_constraint":gpytorch.constraints.constraints.GreaterThan(0.01),
                "outputscale_hyperprior":gpytorch.priors.torch_priors.NormalPrior(loc=2.0,scale=1.0),
                }


        trainer_config = {
                "n_steps" :50 ,
                "report_freq":10,
                "save_best":True,
        }

        optimizer_config = {
                "n_eval":1,
                ### for GIBO
                "n_max":env_appx_config["n_max"],
                "n_info_samples":env_appx_config["n_info"],
                "delta":0.1,
                "normalize_gradient":True if kernel_name == "rbf" else False,
                "standard_deviation_scaling":False,
        }

        return env_config,likelihood_config,kernel_config,optimizer_config,trainer_config