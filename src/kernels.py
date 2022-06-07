import logging
import logging.config
from abc import ABC
from copy import deepcopy

import gpytorch
import torch
### gpytorch 
from gpytorch.kernels import Kernel, MaternKernel, RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors.torch_priors import GammaPrior

logging.config.fileConfig('logging.conf')
logger = logging.getLogger("ShapeLog."+__name__)

from gpytorch.settings import debug

debug._set_state(False) ##hotfix for GridKernel to inherit ScaleKernel

class MyRBFKernel(ScaleKernel):
        
    
    def __init__(self,ard_num_dims,use_ard=True,
                noise_constraint=None,
                noise_hyperprior=None,
                lengthscale_constraint=None,
                lengthscale_hyperprior=None,
                outputscale_constraint=None,
                outputscale_hyperprior=None):

        if use_ard == False :
            ard_num_dims = None
            
        rbf = RBFKernel(
            ard_num_dims=ard_num_dims,
            lengthscale_prior=lengthscale_hyperprior,
            lengthscale_constraint=lengthscale_constraint,
                                )
        
        
        super().__init__(base_kernel=rbf,
            outputscale_prior=outputscale_hyperprior,
            outputscale_constraint=outputscale_constraint,
                                                            ) 
            
        # Initialize lengthscale and outputscale to mean of priors.
        if lengthscale_hyperprior is not None:
            self.base_kernel.lengthscale = lengthscale_hyperprior.mean
        if outputscale_hyperprior is not None:
            self.outputscale = outputscale_hyperprior.mean  
        
    """Toy kernel for warningging"""
    def forward(self,x1,x2,**params):
        

        
        logger.debug(f'x1 {x1.shape} / x2 {x2.shape}')
        rslt = super().forward(x1,x2,**params)
        logger.debug(f'pair rslt {rslt.shape}')
        return rslt



class StateKernel(Kernel):
    
    """Abstract class for a kernel that uses state action pairs metric
    """
    
    def __init__(self,mlp,train_s,
                ard_num_dims,use_ard,
                noise_constraint=None,
                noise_hyperprior=None,
                lengthscale_constraint=None,
                lengthscale_hyperprior=None,
                outputscale_constraint=None,
                outputscale_hyperprior=None):
                
        """
        
        ard_num_dims : in this kernel it's the number of states to take.
        use_ard : whether to give more weight to certain states.
        
        """
        super().__init__()
        self.train_s = None
        self.mlp = None
        self.orig_args = locals().copy()        
        self.n_actions = mlp.n_actions
        self.set_train_data(train_s,mlp)

    def get_kernel_args(self,train_s):
        
        args = self.orig_args.copy()
        
        ### rewrite ard_num_dims
        args["ard_num_dims"] = train_s.shape[0] * args["mlp"].n_actions
        del args["train_s"]
        del args["self"]
        del args["mlp"]
        
        
        return args
        
    
    def build_kernel(self):
        
        raise NotImplementedError
     
    def forward(self,x1,x2,**params):
            
        raise NotImplementedError
    
    
    def set_train_data(self,train_s,mlp):
        
        
        kernel_args = self.get_kernel_args(train_s)
        self.build_kernel(**kernel_args)
        self.__dict__.update(**kernel_args)
        
        self.states = train_s
        self.mlp = mlp
        
    
    
    def append_train_data(self,new_s,mlp):
        
        self.set_train_data(new_s,mlp)
        
    
    def test_policy(self,params_batch,states):
        
        logger.debug(f'mlp :params_batch.shape{params_batch.shape} states.shape {states.shape}')
        actions = self.mlp(params_batch,states) ##[params_batch[:2],n_actions,n_states]
        logger.debug(f'mlp :actions.shape{actions.shape}')
        actions = torch.flatten(actions,start_dim=-2)##[params_batch[:2],n_actions*n_states]
        logger.debug(f'reshape :actions.shape{actions.shape}')
        return actions
    
class LinearStateKernel(StateKernel):
    
    def build_kernel(self,ard_num_dims,use_ard,**kwargs):
        
        self.base_kernel = gpytorch.kernels.LinearKernel()
        self.register_parameter("lengthscales", torch.nn.Parameter(
                                                                (1/ard_num_dims) *torch.ones(ard_num_dims)
                                                                )
                                )
        
        
    def forward(self,x1,x2,**params):
        
      
        
        logger.debug(f'x1 {x1.shape} / x2 {x2.shape}')
        #Evaluate current parameters
        a1 = self.test_policy(x1,self.states)
        a2 = self.test_policy(x2,self.states)   
        logger.debug(f'a1 {a1.shape} a2 {a2.shape} ')
        # Compute pairwise pairwise kernel 
        kernel = self.base_kernel.forward(a1*self.lengthscales, a2*self.lengthscales, **params)
        logger.debug(f'pair kernel {kernel.shape}')
        
        return kernel
        

def setup_kernel(kernel_config,mlp,train_s):
    
    kernel_name = kernel_config.pop("kernel_name")
    
    if kernel_name == "rbf":
        
        kernel = MyRBFKernel(**kernel_config)
    
    elif kernel_name == "grid":
        
        kernel = GridKernel(**kernel_config,mlp=mlp,train_s=train_s)
    
    elif kernel_name == "linearstate":
            
        kernel = LinearStateKernel(**kernel_config,mlp=mlp,train_s=train_s)
    
    else : raise ValueError("Unknown kernel")
    
    return kernel

        
class GridKernel(object):
    pass
        
    
