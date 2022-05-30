import logging
import logging.config

import gpytorch
import torch
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP

from src.kernels import StateKernel, setup_kernel

logging.config.fileConfig('logging.conf')
logger = logging.getLogger("ShapeLog."+__name__)

class MyGP(ExactGP,GPyTorchModel):
    
    _num_outputs = 1
    
    def __init__(self, train_x, train_y,train_s,
                kernel_config,likelihood_config,
                mlp=None,
                ):
        
        self.x_hist = train_x.clone()
        self.y_hist = train_y.clone()
        
        #### maybe add this to kernel

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            **likelihood_config
        )
        
        ## Use fixed (unoptimizable) noise
        # likelihood.noise_covar.noise = 0.01
        # likelihood.noise_covar.raw_noise.requires_grad = False
        
        ######
     
        ExactGP.__init__(self,train_x, train_y, likelihood)
        
        self.N_max = kernel_config.pop('N_max')
        self.mean_module = ConstantMean() ## prior mean = 0
        self.covar_module = setup_kernel(kernel_config,mlp=mlp,train_s=train_s)

        self.N = train_x.shape[0]
        self.D = train_x.shape[1] 

        
        print(f' Using {self.N_max} last points to compute GP')
        
        
    def update_train_data(self,new_x, new_y,new_s,strict=False):
        
        ### concatenate new data
        train_x = torch.cat([self.train_inputs[0], new_x])
        train_y = torch.cat([self.train_targets, new_y])
        
        ExactGP.set_train_data(self,inputs=train_x,targets=train_y,strict=strict)
        self.N = train_x.shape[0]
        
        ### update history 
        self.x_hist = torch.cat([self.x_hist, new_x])
        self.y_hist = torch.cat([self.y_hist, new_y])
        
        ### update state kernels with new states
        if isinstance(self.covar_module,StateKernel):
            self.covar_module.update(new_s)
            

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
  
    def print_hypers(self):
        
        print("##############################")
        print(f'covar_lengthscale max {self.covar_module.base_kernel.lengthscale.max()} / min {self.covar_module.base_kernel.lengthscale.min()}  \
                covar_outputscale {self.covar_module.outputscale.item()} \
                noise {self.likelihood.noise_covar.noise.item()}')
        print("##############################")
                

### FOR GIBO
class DEGP(MyGP):


    def __init__(self,*args,**kwargs):
       
        """Inits GP model with data and a Gaussian likelihood."""

        super().__init__(*args,**kwargs)
        
    def get_L_lower(self):
        """Get Cholesky decomposition L, where L is a lower triangular matrix.

        Returns:
            Cholesky decomposition L.
        """
        return (
            self.prediction_strategy.lik_train_train_covar.root_decomposition()
            .root.evaluate()
            .detach()
        )

    
    def get_KXX_inv(self):
        """Get the inverse matrix of K(X,X).

        Returns:
            The inverse of K(X,X).
        """
        L_inv_upper = self.prediction_strategy.covar_cache.detach()
        return L_inv_upper @ L_inv_upper.transpose(0, 1)

    def _get_KxX_dx(self, x):
        """Computes the analytic derivative of the kernel K(x,X) w.r.t. x.

        Args:
            x: (n x D) Test points.

        Returns:
            (n x D) The derivative of K(x,X) w.r.t. x.
        """
        X = self.train_inputs[0]
        n = x.shape[0]
        K_xX = self.covar_module(x, X).evaluate()
        lengthscale = self.covar_module.base_kernel.lengthscale.detach()
        return (
            -torch.eye(self.D, device=x.device)
            / lengthscale ** 2
            @ (
                (x.view(n, 1, self.D) - X.view(1, self.N, self.D))
                * K_xX.view(n, self.N, 1)
            ).transpose(1, 2)
        )

    def _get_Kxx_dx2(self):
        """Computes the analytic second derivative of the kernel K(x,x) w.r.t. x.

        Args:
            x: (n x D) Test points.

        Returns:
            (n x D x D) The second derivative of K(x,x) w.r.t. x.
        """
        lengthscale = self.covar_module.base_kernel.lengthscale.detach()
        sigma_f = self.covar_module.outputscale.detach()
        return (
            torch.eye(self.D, device=lengthscale.device) / lengthscale ** 2
        ) * sigma_f

    def posterior_derivative(self, x):
        """Computes the posterior of the derivative of the GP w.r.t. the given test
        points x.

        Args:
            x: (n x D) Test points.

        Returns:
            A GPyTorchPosterior.
        """
        if self.prediction_strategy is None:
            self.posterior(x)  # hotfix 
            
        K_xX_dx = self._get_KxX_dx(x)
        mean_d = K_xX_dx @ self.get_KXX_inv() @ self.train_targets
        variance_d = (
            self._get_Kxx_dx2() - K_xX_dx @ self.get_KXX_inv() @ K_xX_dx.transpose(1, 2)
        )
        variance_d = variance_d.clamp_min(1e-9)

        return mean_d, variance_d
            

