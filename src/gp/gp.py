import logging
import logging.config

import gpytorch
import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import InputStandardize
from botorch.models.transforms.outcome import Standardize
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from src.gp.means import MyConstantMean,AdvantageMean

from src.gp.kernels import *
from src.gp.means import *

logging.config.fileConfig('logging.conf')
logger = logging.getLogger("ShapeLog."+__name__)

class MyGP(SingleTaskGP):
    
    _num_outputs = 1
    
    def __init__(self, train_x, train_y,train_s,
                mean_module,covar_module,likelihood,
                ):
        
        self.x_hist = train_x.clone()
        self.y_hist = train_y.clone()
        
        
        ## Use fixed (unoptimizable) noise
        # likelihood.noise_covar.noise = 0.01
        # likelihood.noise_covar.raw_noise.requires_grad = False
        
        ######

        super().__init__(train_X = train_x, 
                         train_Y= train_y.reshape(-1,1),
                        likelihood= likelihood,
                        mean_module =mean_module, ## prior mean = 0
                        covar_module = covar_module,
                        #### NEWW NORMALIZATION
                        # outcome_transform=Standardize(m=train_y.shape[-1]),
                        # input_transform=InputStandardize(d=train_x.shape[-1])
                        )

        self.N = train_x.shape[0]
        self.D = train_x.shape[1] 


    def set_module_data(self,mean,states,transitions):
        
        #if  isinstance(self.covar_module,StateKernel):
            
            
            self.mean_module.set_train_data(mean,states,transitions)
            
            self.covar_module.set_train_data(states)
    
    
    def append_module_data(self,mean,states,transitions):
        
        if  isinstance(self.covar_module,StateKernel):
            
            self.mean_module.append_train_data(mean,states,transitions)
            
            self.covar_module.append_train_data(states)
            
        
            
    
    def set_train_data(self,train_x,train_y,train_s=None,strict=False):
        
        super().set_train_data(inputs=train_x,targets=train_y,strict=strict)
        self.N = train_x.shape[0]
        
        
        
    def append_train_data(self,new_x, new_y,new_s=None,strict=False):
        
        """updates only train_x and train_y (maybe eventually add train_s)
        """
        
        ### concatenate new data
        train_x = torch.cat([self.train_inputs[0], new_x])
        train_y = torch.cat([self.train_targets, new_y])
        
        super().set_train_data(inputs=train_x,targets=train_y,strict=strict)
        self.N = train_x.shape[0]
        
        ### update history 
        self.x_hist = torch.cat([self.x_hist, new_x])
        self.y_hist = torch.cat([self.y_hist, new_y])
        
        ### update state kernels with new states
        if isinstance(self.covar_module,StateKernel) and not(new_s is None):
            self.covar_module.append_train_data(new_s)
    

    def forward(self, x):
        
        
        logger.debug(f'x.shape {x.shape}')
        
        with gpytorch.settings.debug(state=False):
            
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)
        
    def get_best_params(self):
            
        argmax = torch.argmax(self.y_hist)
        best_x = self.x_hist[argmax]
        best_y = self.y_hist[argmax]
        return best_x,best_y
  
    def log_hypers(self,n_samples,):
        
        
        dct = {
            
            "covar_lengthscale mean": self.covar_module.base_kernel.lengthscale.mean(),
            "covar_lengthscale std": self.covar_module.base_kernel.lengthscale.std(),
            "covar_output_scale" : self.covar_module.outputscale.item(),
            "noise": self.likelihood.noise_covar.noise.item(),    
            "Mean MAE" :self.mean_error,
            "R2" : self.R2,
            "R1" : self.R1,
            "mean_on_covar_norm":self.mean_on_covar_norm,
            "mean_on_covar_cos":self.mean_on_covar_cos,
        }
        
        
        self.trainer.log(n_samples,dct)
        
        
    def print_train_mll(self):
        
        self.posterior(self.train_inputs[0][-1].reshape(1,-1))  ## hotfix
        mll = ExactMarginalLogLikelihood(self.likelihood, self)
        train_x = self.train_inputs[0]
        train_y = self.train_targets
        print(f'MLL : {mll(self(train_x),train_y).item()}')
        self.posterior(self.train_inputs[0][-1].reshape(1,-1))  ## hotfix
                    
        
                

### FOR GIBO
class DEGP(MyGP):


    def __init__(self,*args,**kwargs):
       
        """Inits GP model with data and a Gaussian likelihood."""

        super().__init__(*args,**kwargs)
        
    
    def get_KXX_inv(self):
        """Get the inverse matrix of K(X,X).

        Returns:
            The inverse of K(X,X).
        """
        L_inv_upper = self.prediction_strategy.covar_cache.detach()
        return L_inv_upper @ L_inv_upper.transpose(0, 1)

    def _get_KxX_dx(self, theta_t):
        """Computes the analytic derivative of the kernel K(x,X) w.r.t. x.

        Args:
            x: (n x D) Test points.

        Returns:
            (n x D) The derivative of K(x,X) w.r.t. x.
        """
        X_hat = self.train_inputs[0]
        jacobs = torch.autograd.functional.jacobian(func=lambda theta : self.covar_module(theta,X_hat).evaluate(),
                                                    inputs=(theta_t))
        K_θX_dθ = jacobs.sum(dim=2).transpose(1,2)

        return K_θX_dθ
        

    def _get_Kxx_dx2(self,theta_t):
        """Computes the analytic second derivative of the kernel K(x,x) w.r.t. x.

        Args:
            x: (n x D) Test points.

        Returns:
            (n x D x D) The second derivative of K(x,x) w.r.t. x.
        """

        
        theta = theta_t.clone()
        theta.requires_grad = True
        theta_t2 = theta_t.clone().detach() ## hotfix otherwise 0 hessian
        ## this might be a cause of error, try to find method to compute using k(theta,theta)
        hessian = torch.autograd.functional.hessian(func=lambda theta : self.covar_module(theta,theta_t2).evaluate(),
                                                    inputs=(theta)).squeeze()
    
        return -hessian

            
    
    def get_Mx_dx(self,theta_t):
        
        if  isinstance(self.mean_module,MyConstantMean):
            
                return torch.zeros_like(theta_t)
        
        else : 
                
                #tmp = torch.tensor(theta_t,requires_grad=True)
                theta_t.requires_grad = True
                out = self.mean_module.call2(theta_t)
                Mx_dx =  torch.autograd.grad(out,theta_t)
                theta_t.requires_grad = False
                
                return Mx_dx[0]
        
      
        
    def posterior_derivative(self,theta_t):
        """Computes the posterior of the derivative of the GP w.r.t. the given test
        points x.

        Args:
            x: (n x D) Test points.

        Returns:
            A GPyTorchPosterior.
        """
        if self.prediction_strategy is None:
            self.posterior(theta_t)  # hotfix 
            
        K_xX_dx = self._get_KxX_dx(theta_t)
        Kxx_dx2 = self._get_Kxx_dx2(theta_t)
        KXX_inv = self.get_KXX_inv()
        
        with torch.enable_grad():
            if  isinstance(self.mean_module,AdvantageMean):
                
                self.mean_module.fit_critic()
                
            
        M_x = self.mean_module(self.train_inputs[0])
        
        with torch.enable_grad():
            mean_grad = self.get_Mx_dx(theta_t)


        covar_grad = K_xX_dx @ KXX_inv @ (self.train_targets- M_x)

        
        # print(f'mean grad {mean_grad}')
        # print(f'covar_gad {covar_grad}')
        # print(f'diff {(self.train_targets- M_x)}')
        # print(f' K_xX_dx {K_xX_dx} ')
        mean_d =   mean_grad + covar_grad
        
        variance_d =  Kxx_dx2 - K_xX_dx @ KXX_inv @ K_xX_dx.transpose(1, 2)

    

        
        ### Variance can be negative (although diagonal should be positive for a PSD matrix)
        #variance_d = variance_d.clamp_min(1e-9)
        
        
        
        y = self.train_targets 
        y_bar =y.mean()
        
        self.mean_error = torch.mean(torch.abs(self.train_targets- M_x))
        self.R2 = 1 - torch.sum((M_x-y)**2)/torch.sum((y_bar-y)**2)
        self.R1 = 1 - torch.sum(torch.abs(M_x-y))/torch.sum(torch.abs(y_bar-y))

        self.mean_on_covar_cos = torch.nn.functional.cosine_similarity(mean_grad,covar_grad)
        self.mean_on_covar_norm = torch.norm(mean_grad)/torch.norm(covar_grad)

        return mean_d, variance_d
            
 