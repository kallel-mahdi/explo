import logging
import logging.config

import gpytorch
import torch
### gpytorch 
from gpytorch.kernels import Kernel, MaternKernel, RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors.torch_priors import GammaPrior
from copy import deepcopy

logging.config.fileConfig('logging.conf')
logger = logging.getLogger("ShapeLog."+__name__)



from gpytorch.settings import debug
debug._set_state(False) ##hotfix for GridKernel to inherit ScaleKernel

class MyRBFKernel(ScaleKernel):
        
    
    def __init__(self,ard_num_dims,use_ard,
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
        
    def forward(self,x1,x2,**params):
        

        
        logger.debug(f'x1 {x1.shape} / x2 {x2.shape}')
        rslt = super().forward(x1,x2,**params)
        logger.debug(f'pair rslt {rslt.shape}')
        return rslt


class StateKernel(MyRBFKernel):
    
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
        
        self.args = deepcopy(locals())
        self.n_actions = mlp.n_actions
        self.mlp = mlp
        self.reset(train_s)
        
    
    def reset(self,train_s):
        
        args = self.args.copy()
        del args["ard_num_dims"]
        del args["train_s"]
        del args["self"]
        del args["mlp"]
        print(args)
        
        
        ard_num_dims = train_s.shape[0] * self.n_actions
        
        super().__init__(ard_num_dims,**args)
        
        self.states = train_s
    
        

    def test_policy(self,params_batch,states):
        
        logger.debug(f'mlp :params_batch.shape{params_batch.shape} states.shape {states.shape}')
        actions = self.mlp(params_batch,states) ##[params_batch[:2],n_actions,n_states]
        logger.debug(f'mlp :actions.shape{actions.shape}')
        actions = torch.flatten(actions,start_dim=-2)##[params_batch[:2],n_actions*n_states]
        logger.debug(f'reshape :actions.shape{actions.shape}')
        return actions
    
    def forward(self,x1,x2,**params):
        
        
        logger.debug(f'x1 {x1.shape} / x2 {x2.shape}')
        
        #Evaluate current parameters
        actions1 = self.test_policy(x1,self.states)
        actions2 = self.test_policy(x2,self.states)
        logger.debug(f'actions1 {actions1.shape} actions2 {actions2.shape} ')
        # Compute pairwise pairwise kernel 
        kernel = super().forward(actions1, actions2, **params)
        logger.debug(f'pair kernel {kernel.shape}')
        
        return kernel 
    
        
    def update_states(self,new_s):
        
        raise NotImplementedError
    
    def update(self,new_s):
        
        raise NotImplementedError
    
    
class GridKernel(StateKernel):
    
    """ a statekernel that uses a grid over state space to 
    compute the kernel
    """
    
    def __init__(self,*args,**kwargs):
        
        super(GridKernel, self).__init__(*args,**kwargs)

    def get_grid(self,low,high,samples_per_dim):
        
        
        state_dims = low.shape[0]
        points = [torch.linspace(low[i],high[i],samples_per_dim) 
                    for i in range(state_dims)]
        grid = torch.meshgrid(*points)
        grid = torch.stack(grid)
        grid = torch.flatten(grid,start_dim=1).T ## [n_states,state_dim]
        
        logger.debug(f' grid shape {grid.shape}')
        
        return grid
    
    
    def generate_data(self,low,high,n_samples):
        
        logger.debug(f'generating {n_samples} uniformly for grid')
        U = torch.distributions.Uniform(low,high)
        data = U.sample(sample_shape=[n_samples])
        
        return data
    
    def update_states(self,new_s):
        
        self.high,_= torch.max(new_s,dim=0)
        self.low,_= torch.min(new_s,dim=0)
        n_samples = int(self.ard_num_dims/self.n_actions)
        states = self.generate_data(self.low,self.high,n_samples)
        n_states = states.shape[0]
        self.states = states
        
        return n_states
        
    
    def update(self,new_s):
        
        tmp_buff = torch.cat([self.states, new_s])
        high,_= torch.max(tmp_buff,dim=0)
        low,_= torch.min(tmp_buff,dim=0)
        
        logger.debug(f'Buffer shape {tmp_buff.shape}')
        
        ### update only if new boundaries
        if any(high>self.high) or any(low<self.low):
            self.update_states(tmp_buff)


class MyMaternKernel(Kernel):
    
    def __init__(self,ard_num_dims,use_ard=False):
        
        super().__init__()
        if not use_ard: ard_num_dims = None
        self.kernel = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=ard_num_dims,
                    lengthscale_prior=GammaPrior(3.0, 6.0)),
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
       
    """Toy kernel for warningging"""
    def forward(self,x1,x2,**params):
        
        logger.debug(f'x1 {x1.shape} / x2 {x2.shape}')
        kernel = self.kernel.forward(x1,x2,**params)
        logger.debug(f'pair kernel {kernel.shape}')
        return kernel

def setup_kernel(kernel_config,mlp,train_s):
    
    kernel_name = kernel_config.pop("kernel_name")
    
    if kernel_name == "rbf":
        
        kernel = MyRBFKernel(**kernel_config)
    
    elif kernel_name == "grid":
        
        kernel = GridKernel(**kernel_config,mlp=mlp,train_s=train_s)
    
    else : raise ValueError("Unknown kernel")
    
    return kernel

        
        
        
    
