import torch
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP

#######
from src.kernels import *
import gpytorch

class MyGP(ExactGP,GPyTorchModel):
    
    _num_outputs = 1
    
    def __init__(self, train_x, train_y,train_s,
                lengthscale_constraint=None,
                lengthscale_hyperprior=None,
                outputscale_constraint=None,
                outputscale_hyperprior=None,
                noise_constraint=None,
                noise_hyperprior=None,
                ard_num_dims=None,
                N_max=None,
                prior_mean=0,
                mlp=None,use_ard=False):
        
        
        
        # self.covar_module = self.setup_kernel(kernel_name,ard_num_dims,use_ard,
        #                                       mlp,train_s)
        
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=noise_constraint, noise_prior=noise_hyperprior
        )
        
        ### Do not optimize noise
        likelihood.noise_covar.noise = 0.01
        likelihood.noise_covar.raw_noise.requires_grad = False
        ###
        
        ExactGP.__init__(self,train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=ard_num_dims,
                lengthscale_prior=lengthscale_hyperprior,
                lengthscale_constraint=lengthscale_constraint,
            ),
            outputscale_prior=outputscale_hyperprior,
            outputscale_constraint=outputscale_constraint,
        )
        # Initialize lengthscale and outputscale to mean of priors.
        if lengthscale_hyperprior is not None:
            self.covar_module.base_kernel.lengthscale = lengthscale_hyperprior.mean
        if outputscale_hyperprior is not None:
            self.covar_module.outputscale = outputscale_hyperprior.mean
        
        self.N = train_x.shape[0]
        self.D = train_x.shape[1] 
        self.N_max = N_max  
        
        
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
            

