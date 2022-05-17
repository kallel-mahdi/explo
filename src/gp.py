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
                 kernel=None,mlp=None):
        
        ExactGP.__init__(self,train_x, train_y, likelihood)
        
        self.mean_module = ConstantMean()
        
        if kernel is None:
            self.covar_module = MyKernel()
        else :
            self.covar_module = kernel(mlp,train_s)
            
    def update_train_data(self,new_x, new_y,new_s,strict=False):
        
        train_x = torch.cat([self.train_inputs[0], new_x])
        train_y = torch.cat([self.train_targets, new_y])
        ExactGP.set_train_data(self,inputs=train_x,targets=train_y,strict=strict)
        
        
        ### update state kernels with new states
        if isinstance(self.covar_module,StateKernel):
            self.covar_module.update(new_s)
        
    def forward(self, x):
        
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
