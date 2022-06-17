import logging
import logging.config
from abc import ABC
from copy import deepcopy

import gpytorch
import torch
### gpytorch 
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel,LinearKernel,MaternKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior


logging.config.fileConfig('logging.conf')
logger = logging.getLogger("ShapeLog."+__name__)

from gpytorch.settings import debug

debug._set_state(False) ##hotfix for GridKernel to inherit ScaleKernel

class MyRBFKernel(ScaleKernel):
        
    
    def __init__(self,ard_num_dims,use_ard,
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
        
        
        ScaleKernel.__init__(self,base_kernel=rbf,
            outputscale_prior=outputscale_hyperprior,
            outputscale_constraint=outputscale_constraint,
                                                            ) 
            
        # Initialize lengthscale and outputscale to mean of priors.
        if lengthscale_hyperprior is not None:
            self.base_kernel.lengthscale = lengthscale_hyperprior.mean
        if outputscale_hyperprior is not None:
            self.outputscale = outputscale_hyperprior.mean  
        
    def forward(self,x1,x2,**params):
        
        logger.debug(f'x1 {x1.shape} / x2 {x2.shape}')
        rslt = super().forward(x1,x2,**params)
        logger.debug(f'pair rslt {rslt.shape}')
        return rslt



class MyMaternKernel(ScaleKernel):
    
    def __init__(self,ard_num_dims,use_ard,**kwargs):
    
        if use_ard == False :
                ard_num_dims = None
                
        ScaleKernel.__init__(self,
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=ard_num_dims,
                    #batch_shape=self._aug_batch_shape,
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                ),
                #batch_shape=self._aug_batch_shape,
                outputscale_prior=GammaPrior(2.0, 0.15),
            )


class StateKernel:
    
    """Abstract class for a kernel that uses state action pairs metric
    """
    
    def __init__(self,mlp,train_s,
                **kernel_config
                ):
                
        """
        
        use_ard : whether to give more weight to certain states.
        ard_num_dims : in this kernel it's the number of states to take.
        
        """
      
        
        self.states = train_s
        self.n_actions = mlp.n_actions
        self.kernel_config = kernel_config
        self.set_train_data(train_s,mlp)
        self.mlp = mlp ## this must be set after updating kernel
    
    def set_train_data(self,train_s,mlp):
        """ sometimes we need to reset the states used by the kernel
        This usually requires re insantiating the base kernel (RBF or Linear ..) """
            
        self.kernel_config["ard_num_dims"] = train_s.shape[0] * mlp.n_actions
        self.build_kernel(**self.kernel_config)
        self.states = train_s
        ## removing this works for linearstate but not rbf kernel
        ## might be worth investigation
        self.mlp = mlp 


    def append_train_data(self,new_s,mlp):
        
        self.set_train_data(new_s,mlp)

    def build_kernel(self):
        
        raise NotImplementedError
     
    def forward(self,x1,x2,**params):
            
        raise NotImplementedError
    
    
        
    
    def run_parameters(self,params_batch,states):
        
        logger.debug(f'mlp :params_batch.shape{params_batch.shape} states.shape {states.shape}')
        actions = self.mlp(params_batch,states) ##[params_batch[:2],n_actions,n_states]
        logger.debug(f'mlp :actions.shape{actions.shape}')
        actions = torch.flatten(actions,start_dim=-2)##[params_batch[:2],n_actions*n_states]
        logger.debug(f'reshape :actions.shape{actions.shape}')
        return actions
    


class LinearStateKernel(LinearKernel,StateKernel):
    
    
    def __init__(self,**kwargs):
        
        StateKernel.__init__(self,**kwargs)
    
    def build_kernel(self,ard_num_dims,use_ard,**kwargs):
        
        LinearKernel.__init__(self,
                              variance_constriant = gpytorch.constraints.constraints.GreaterThan(0.1))
        
        if use_ard: 
            
            self.register_parameter("lengthscales", torch.nn.Parameter(
                                                                    (1/ard_num_dims) * torch.ones(ard_num_dims)
                                                                    )
                                    )
        
        else : 
            
            self.lengthscales = (1/ard_num_dims) * torch.ones(ard_num_dims)
            self.lengthscales.requires_grad = False
              
        def forward(self,x1,x2,**params):
            
            
            logger.debug(f'x1 {x1.shape} / x2 {x2.shape}')
            #Evaluate current parameters
            a1 = self.run_parameters(x1,self.states)
            a2 = self.run_parameters(x2,self.states)   
            logger.debug(f'a1 {a1.shape} a2 {a2.shape} ')
            # Compute pairwise pairwise kernel 
            kernel = super().forward(self.lengthscales* a1,a2, **params)
            logger.debug(f'pair kernel {kernel.shape}')
            
            return kernel
            
            
class RBFStateKernel(MyRBFKernel,StateKernel):
    
        def __init__(self,**kwargs):
            
            StateKernel.__init__(self,**kwargs)
    
        def build_kernel(self,ard_num_dims,use_ard,**kwargs):
                        
            MyRBFKernel.__init__(self,ard_num_dims,use_ard,**kwargs)
            
            
            self.base_kernel.lengthscale = torch.sqrt(torch.Tensor([ard_num_dims]))
            self.outputscale = torch.Tensor([1.])
            
            #print(f'self outputscale {self.outputscale.requires_grad}')
        
        def forward(self,x1,x2,**params):
                
            logger.debug(f'x1 {x1.shape} / x2 {x2.shape}')
            #Evaluate current parameters
            a1 = self.run_parameters(x1,self.states)
            a2 = self.run_parameters(x2,self.states)   
            logger.debug(f'a1 {a1.shape} a2 {a2.shape} ')
            # Compute pairwise pairwise kernel 
            kernel = super().forward(a1, a2, **params)
            logger.debug(f'pair kernel {kernel.shape}')
            
            return kernel
        


class MaternStateKernel(MyMaternKernel,StateKernel):
    
        def __init__(self,**kwargs):
            
            StateKernel.__init__(self,**kwargs)
    
        def build_kernel(self,ard_num_dims,use_ard,**kwargs):
                        
            MyMaternKernel.__init__(self,ard_num_dims,use_ard,**kwargs)
            #self.base_kernel.lengthscale = torch.sqrt(torch.Tensor([ard_num_dims]))
            #self.base_kernel.lengthscale.requires_grad = False
        
        def forward(self,x1,x2,**params):
                
            logger.debug(f'x1 {x1.shape} / x2 {x2.shape}')
            #Evaluate current parameters
            a1 = self.run_parameters(x1,self.states)
            a2 = self.run_parameters(x2,self.states)   
            logger.debug(f'a1 {a1.shape} a2 {a2.shape} ')
            # Compute pairwise pairwise kernel 
            kernel = super().forward(a1, a2, **params)
            logger.debug(f'pair kernel {kernel.shape}')
            
            return kernel
            
        
class GridKernel(object):
    pass
        
    
