from gpytorch.kernels import Kernel,RBFKernel
import logging
import logging.config

logging.config.fileConfig('logging.conf')
logger = logging.getLogger("ShapeLog."+__name__)

class MyKernel(RBFKernel):
   
    """Toy kernel for warningging"""
    def forward(self,x1,x2,**params):
        
        logger.debug(f'x1 {x1.shape} / x2 {x2.shape}')
        kernel = super().forward(x1,x2,**params)
        logger.debug(f'pair kernel {kernel.shape}')
        return kernel


class StateKernel(Kernel):
    
    """Abstract class for a kernel that uses state action pairs metric
    """
    
    def __init__(self,mlp,train_s):
        
        super().__init__()
        self.mlp = mlp
        ### set rbf_module with appropriate ard dims
        n_states = self.update_states(train_s)
        
        #self.rbf_module = ScaleKernel(RBFKernel(ard_num_dims=n_states))
        self.rbf_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=None,
                    lengthscale_prior=GammaPrior(3.0, 6.0)),

                outputscale_prior=GammaPrior(2.0, 0.15),
            )

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
        kernel = self.rbf_module(actions1, actions2, **params)
        logger.debug(f'pair kernel {kernel.shape}')
        
        return kernel 
    
    def update(self,new_s):
        
        raise NotImplementedError