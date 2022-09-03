import logging
import logging.config

import torch
### gpytorch 
from gpytorch.kernels import RBFKernel, ScaleKernel

logging.config.fileConfig('logging.conf')
logger = logging.getLogger("ShapeLog."+__name__)

from gpytorch.settings import debug

debug._set_state(False) ##hotfix to allow input dim and ard_dim to have different dimensions

class MyRBFKernel(ScaleKernel):
        
    
    def __init__(self,ard_num_dims,use_ard,
                lengthscale_constraint=None,
                lengthscale_hyperprior=None,
                outputscale_constraint=None,
                outputscale_hyperprior=None,
                mlp = None, ## unused
                
                ):
        
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

        
        ## Used only for logging purpouses
        self.states = None 

            
        
    def forward(self,x1,x2,**params):
        
        rslt = super().forward(x1,x2,**params)
        
        return rslt

    
        
    def set_train_data(self,new_s):
        
        self.states = new_s
    
    def append_train_data(self,new_s):

        pass
        

class StateKernel:
    
    """Abstract class for a kernel that uses state action pairs metric
    """
    
    def __init__(self,train_s,
                **kernel_config
                ):
                
        """
        
        use_ard : whether to give more weight to certain states.
        ard_num_dims : in this kernel it's the number of states to take.
        
        """
      
        self.kernel_config = kernel_config
        self.set_train_data(train_s)
        
        ## intialised by set_train_data
        
        self.mlp = None 
        self.states = None
        self.n_actions = None 
  

    def build_kernel(self,**kwargs):
        
        raise NotImplementedError
     
    def forward(self,x1,x2,**params):
            
        raise NotImplementedError
 
    def run_parameters(self,params_batch,states):
        
        actions = self.mlp(states,params_batch) ##[params_batch[:2],n_actions,n_states]    
        
        return actions
    
            
class RBFStateKernel(MyRBFKernel,StateKernel):
    
        def __init__(self,**kwargs):
            
            StateKernel.__init__(self,**kwargs)
    
        def build_kernel(self,use_ard,
                         mlp,train_s,**kwargs):
            
    
            ard_num_dims  = train_s.shape[0]
            MyRBFKernel.__init__(self,ard_num_dims,use_ard,**kwargs)
            
            self.base_kernel.lengthscale = torch.Tensor([1.])
            self.outputscale = torch.Tensor([1.])
            self.ard_num_dims = ard_num_dims
            self.states = train_s
            self.mlp = mlp
        
        def forward(self,x1,x2,**params):
                
            #Evaluate current parameters
            a1 = self.run_parameters(x1,self.states)
            a2 = self.run_parameters(x2,self.states) 

            n_actions,n_states = a1.shape[-2],a1.shape[-1]

            # Compute pairwise kernel 
            norm = torch.sqrt(torch.Tensor([self.ard_num_dims]))
            
            kernel = super().forward(a1[:,0,:]/norm, a2[:,0,:]/norm, **params)

            for i in range(1,n_actions):

                kernel *= super().forward(a1[:,i,:]/norm, a2[:,i,:]/norm, **params)

        
            return kernel
        

        def set_train_data(self,train_s):

            """ sometimes we need to reset the states used by the kernel
            This usually requires re insantiating the base kernel (RBF or Linear ..) """
                
            self.build_kernel(**self.kernel_config,train_s=train_s)
            
            
        def append_train_data(self,new_s):
            
            pass
            
    
