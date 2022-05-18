import torch
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP

#######
from src.kernels import *

class MyGP(ExactGP,GPyTorchModel):
    
    _num_outputs = 1
    
    def __init__(self, train_x, train_y,train_s, likelihood,
                 kernel_name,mlp=None,use_ard=False):
        
        ExactGP.__init__(self,train_x, train_y, likelihood)
        
        ard_num_dims = train_x.shape[-1]
        self.mean_module = ConstantMean()
        self.covar_module = self.setup_kernel(kernel_name,ard_num_dims,use_ard,
                                              mlp,train_s)
        
    def update_train_data(self,new_x, new_y,new_s,strict=False):
        
        train_x = torch.cat([self.train_inputs[0], new_x])
        train_y = torch.cat([self.train_targets, new_y])
        ExactGP.set_train_data(self,inputs=train_x,targets=train_y,strict=strict)
        self.N = train_x.shape[0]
        
        ### update state kernels with new states
        if isinstance(self.covar_module,StateKernel):
            self.covar_module.update(new_s)
        
    def forward(self, x):
        
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
    def get_best_params(self):
            
        argmax = torch.argmax(self.train_targets)
        best_x = self.train_inputs[0][argmax]
        best_y = self.train_targets[argmax]
        return best_x,best_y
    
    
    def setup_kernel(self,kernel_name,ard_num_dims,use_ard,mlp,train_s):
        
        if kernel_name == "rbf":
            kernel = MyRBFKernel(ard_num_dims,use_ard)
        elif kernel_name == "matern":
            kernel = MyMaternKernel(ard_num_dims,use_ard)    
        elif kernel_name == "grid":
            kernel = GridKernel(mlp,train_s)
            
        else: raise ValueError("Unknown kernel")
        
        return kernel
            

