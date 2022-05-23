import logging
import logging.config
import torch
import gpytorch
### gpytorch 
from gpytorch.kernels import Kernel, MaternKernel, RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors.torch_priors import GammaPrior

logging.config.fileConfig('logging.conf')
logger = logging.getLogger("ShapeLog."+__name__)

class StateKernel(Kernel):
    
    """Abstract class for a kernel that uses state action pairs metric
    """
    
    def __init__(self,mlp,train_s,kernel,
                ):
        
        super().__init__()
        self.mlp = mlp
        ### set kernel with appropriate ard dims
        n_states = self.update_states(train_s)
        
        ### chose kernel
        
        #self.kernel = ScaleKernel(RBFKernel())
        self.kernel = kernel

    def test_policy(self,params_batch,states):
        
        logger.debug('mlp :params_batch.shape{params_batch.shape}')
        actions = self.mlp(params_batch,states)
        logger.debug('mlp :actions.shape{actions.shape}')
        # first_dims = params_batch.shape[:-1]
        # last_dims = actions.shape[-2:]
        # actions = actions.reshape(*first_dims,*last_dims)
        actions = torch.flatten(actions,start_dim=-2)
        logger.debug('reshape :actions.shape{actions.shape}')
        return actions
    
    def forward(self,x1,x2,**params):
        
        logger.debug(f'x1 {x1.shape} / x2 {x2.shape}')
        
        #Evaluate current parameters
        actions1 = self.test_policy(x1,self.states)
        actions2 = self.test_policy(x2,self.states)
        logger.debug(f'actions1 {actions1.shape} actions2 {actions2.shape} ')
        # Compute pairwise pairwise kernel 
        kernel = self.kernel(actions1, actions2, **params)
        logger.debug(f'pair kernel {kernel.shape}')
        
        return kernel 
    
    def update(self,new_s):
        
        raise NotImplementedError
    
    
class GridKernel(StateKernel):
    
    """ a statekernel that uses a grid over state space to 
    compute the kernel
    """

    def get_grid(self,low,high,samples_per_dim):
        
        
        state_dims = low.shape[0]
        points = [torch.linspace(low[i],high[i],samples_per_dim) 
                    for i in range(state_dims)]
        grid = torch.meshgrid(*points)
        grid = torch.stack(grid)
        grid = torch.flatten(grid,start_dim=1).T ## [n_states,state_dim]
        
        logger.debug(f' grid shape {grid.shape}')
        
        return grid
    
    def update_states(self,new_s):
        
        self.high,_= torch.max(new_s,dim=0)
        self.low,_= torch.min(new_s,dim=0)
        states = self.get_grid(self.low,self.high,
                                    samples_per_dim=5)
        
        n_states = states.shape[0]
        self.states = states
        return n_states
        
        #print(f'observation box : \n low {self.low} \n high :{self.high} \n grid shape {self.states.shape}')
    
    def update(self,new_s):
        
        
        tmp_buff = torch.cat([self.states, new_s])
        high,_= torch.max(tmp_buff,dim=0)
        low,_= torch.min(tmp_buff,dim=0)
        
        logger.debug(f'Buffer shape {tmp_buff.shape}')
        
        ### update only if be
        if any(high>self.high) or any(low<self.low):
            self.update_states(tmp_buff)


class MyRBFKernel(Kernel):
    
    def __init__(self,ard_num_dims,use_ard=False):
        
        super().__init__()
        if not use_ard: ard_num_dims = None
        self.kernel = ScaleKernel(RBFKernel(ard_num_dims=ard_num_dims))
        
       
    """Toy kernel for warningging"""
    def forward(self,x1,x2,**params):
        
        logger.debug(f'x1 {x1.shape} / x2 {x2.shape}')
        kernel = self.kernel.forward(x1,x2,**params)
        logger.debug(f'pair kernel {kernel.shape}')
        return kernel


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




class MyKernel(Kernel):
    
    def __init__(self,ard_num_dims,use_ard=False):
        
        super().__init__()
        
        self.covar_module = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(
            ard_num_dims=ard_num_dims,
            lengthscale_prior=lengthscale_hyperprior,
            lengthscale_constraint=lengthscale_constraint,
                                ),
            outputscale_prior=outputscale_hyperprior,
            outputscale_constraint=outputscale_constraint,
                                                        )   
    
        
        
        
    
    