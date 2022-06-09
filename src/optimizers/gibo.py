import logging
import logging.config

import botorch
import gpytorch
import torch
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from src.kernels import *  # # delte later
from src.optimizers.helpers import my_fit_gpytorch_model
#from torch.optim import LBFGS ## Full pytorch LBFGS implementation

logging.config.fileConfig('logging.conf')
logger = logging.getLogger("ShapeLog."+__name__)



class GradientInformation(botorch.acquisition.AnalyticAcquisitionFunction):
    '''Acquisition function to sample points for gradient information.

    Attributes:
        model: Gaussian process model that supplies the Jacobian (e.g. DerivativeExactGPSEModel).
    '''

    def __init__(self, model):
        '''Inits acquisition function with model.'''
        super().__init__(model)

    def update_theta_i(self, theta_i):
        '''Updates the current parameters.

        This leads to an update of K_xX_dx.

        Args:
            theta_i: New parameters.
        '''
        if not torch.is_tensor(theta_i):
            theta_i = torch.tensor(theta_i)
        self.theta_i = theta_i
        self.update_K_xX_dx()

    def update_K_xX_dx(self):
        '''When new x is given update K_xX_dx.'''
        # Pre-compute large part of K_xX_dx.
        X = self.model.train_inputs[0]
        x = self.theta_i.view(-1, self.model.D)
        self.K_xX_dx_part = self._get_KxX_dx(x, X)

    def K_θX(self,theta_t,X_hat):
        
        rslt = self.model.covar_module(theta_t,X_hat).evaluate()
        
        return rslt

    def _get_KxX_dx(self, theta_t, X_hat) :
        '''Computes the analytic derivative of the kernel K(x,X) w.r.t. x.

        Args:
            x: (n x D) Test points.

        Returns:
            (n x D) The derivative of K(x,X) w.r.t. x.
        '''
        
        jacobs = torch.autograd.functional.jacobian(func=lambda theta : self.K_θX(theta,X_hat),inputs=(theta_t))
        K_θX_dθ = jacobs.sum(dim=2).transpose(1,2)

        return K_θX_dθ

    # TODO: nicer batch-update for batch of thetas.
    @botorch.utils.transforms.t_batch_mode_transform(expected_q=1)
    def forward(self, thetas) :
        
        '''Evaluate the acquisition function on the candidate set thetas.

        Args:
            thetas: A (b) x D-dim Tensor of (b) batches with a d-dim theta points each.

        Returns:
            A (b)-dim Tensor of acquisition function values at the given theta points.
        '''
        sigma_n = self.model.likelihood.noise_covar.noise
        D = self.model.D
        X = self.model.train_inputs[0]
        x = self.theta_i.view(-1, D)

        variances = []
        for theta in thetas:
            theta = theta.view(-1, D)
            
            X_hat = torch.cat([X,theta])
            K_XX = self.model.covar_module(X_hat,X_hat).evaluate() + sigma_n * torch.eye(X_hat.shape[0])
            K_XX_inv = torch.linalg.inv(K_XX)

            # get K_xX_dx
            K_xθ_dx = self._get_KxX_dx(x, theta)
            K_xX_dx = torch.cat([self.K_xX_dx_part, K_xθ_dx], dim=-1)

            # Compute_variance.
            variance_d = -K_xX_dx @ K_XX_inv @ K_xX_dx.transpose(1, 2)
            variances.append(torch.trace(variance_d.view(D, D)).view(1))

        return -torch.cat(variances, dim=0)




class GIBOptimizer(object):
        
    def __init__(self,model,n_eval,
                n_max,n_info_samples,
                normalize_gradient,standard_deviation_scaling,
                delta):

        gradInfo = GradientInformation(model)
        theta_i = model.train_inputs[0][-1].reshape(1,-1)
        params_history = [theta_i.clone()]
        len_params = theta_i.shape[-1]
        optimizer_torch = torch.optim.SGD([theta_i], lr=0.5)
        
        
        self.__dict__.update(locals())
        
        print(f' Gibo will use {self.n_max} last points to fit GP and {self.n_info_samples} info samples')
        
    def optimize_information(self,objective_env,model,bounds):
    
        ## Optimize gradient information locally
        for _ in range(self.n_info_samples):

            model.posterior(self.theta_i)  ## hotfix

            # Optimize acquistion function and get new observation.
            new_x, _ = botorch.optim.optimize_acqf(
                acq_function=self.gradInfo,
                bounds=bounds,
                q=1,  # Analytic acquisition function.
                num_restarts=5,
                raw_samples=64,
                options={'nonnegative': True, 'batch_limit': 5},
                return_best_only=True,
                sequential=False)
            
            # Update training points.
            new_y,new_s = objective_env(new_x,self.n_eval)
            model.append_train_data(new_x,new_y, strict=False) ## right now we do not add new_s for info
            model.posterior(self.theta_i)  ## hotfix
            self.gradInfo.update_K_xX_dx()

    def one_gradient_step(self,model,theta_i):
        
        ## compute gradients manually
        with torch.no_grad():
          
            self.optimizer_torch.zero_grad()
            mean_d, variance_d = model.posterior_derivative(theta_i)
            params_grad = -mean_d.view(1, self.len_params)

            if self.normalize_gradient:
                lengthscale = model.covar_module.base_kernel.lengthscale.detach()
                params_grad = torch.nn.functional.normalize(params_grad) * lengthscale

            if self.standard_deviation_scaling:
                params_grad = params_grad / torch.diag(variance_d.view(self.len_params, self.len_params))

            theta_i.grad = params_grad  # set gradients
            self.optimizer_torch.step()                  

    def step(self,model,objective_env):
   
        theta_i = self.theta_i
 
        # Evaluate current parameters
        new_y,new_s = objective_env(theta_i,self.n_eval)
        model.append_train_data(theta_i,new_y,new_s, strict=False)
        self.gradInfo.update_theta_i(theta_i)
        
        # Only optimize model hyperparameters if N >= n_max.
        if (model.N >= self.n_max): 
            
            # Adjust hyperparameters
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)
            
            # Restrict data to only recent points
            last_x = model.train_inputs[0][-self.n_max:]
            last_y = model.train_targets[-self.n_max:]
            model.set_train_data(last_x,last_y,strict=False)
            model.posterior(self.theta_i)  ## hotfix
            self.gradInfo.update_K_xX_dx() ## hotfix
        
        # Sample locally to optimize gradient information
        bounds = torch.tensor([[-self.delta], [self.delta]]) + theta_i
        self.optimize_information(objective_env,model,bounds)
        
        # Take one step in direction of the gradient
        self.one_gradient_step(model, theta_i)
        
        # Add new theta_i to history 
        self.params_history.append(theta_i.clone())
        
