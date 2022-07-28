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
    
    def K_xX(self,theta_t,X_hat):
            
        rslt = self.model.covar_module(theta_t,X_hat).evaluate()
        
        return rslt

    def update_K_xX_dx(self):
        
        '''When new x is given update K_xX_dx.'''
        # Pre-compute large part of K_xX_dx.
        X = self.model.train_inputs[0]
        x = self.theta_i.view(-1, self.model.D)
        self.K_xX_dx_part = self._get_KθX_dθ(x, X)

  

    def _get_KθX_dθ(self, theta_t, X_hat) :
        '''Computes the analytic derivative of the kernel K(x,X) w.r.t. x.

        Args:
            x: (n x D) Test points.

        Returns:
            (n x D) The derivative of K(x,X) w.r.t. x.
        '''
        
        jacobs = torch.autograd.functional.jacobian(func=lambda theta : self.K_xX(theta,X_hat),inputs=(theta_t))
        KθX_dθ = jacobs.sum(dim=2).transpose(1,2)

        return KθX_dθ

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
        ## does this include theta_i???
        X = self.model.train_inputs[0] 
        x = self.theta_i.view(-1, D)
        variances = []
        
        for theta in thetas:
            
            theta = theta.view(-1, D)

            X_hat = torch.cat([X,theta])
            K_XX = self.model.covar_module(X_hat,X_hat).evaluate() + sigma_n * torch.eye(X_hat.shape[0])
            K_XX_inv = torch.linalg.inv(K_XX)

            # get K_xX_dx
            K_xθ_dx = self._get_KθX_dθ(x, theta)
            K_xX_dx = torch.cat([self.K_xX_dx_part, K_xθ_dx], dim=-1)

            # Compute_variance.
            variance_d = -K_xX_dx @ K_XX_inv @ K_xX_dx.transpose(1, 2)
            variance_d = variance_d.squeeze()
            variances.append(torch.trace(variance_d).view(1))

        return -torch.cat(variances, dim=0)



class GIBOptimizer(object):
        
    def __init__(self,agent,model,
                n_eval,n_max,n_info_samples,
                normalize_gradient,standard_deviation_scaling,
                delta):

        gradInfo = GradientInformation(model)
        theta_i = agent._actor_approximator.model.network.default_weights.data
        params_history = [theta_i.clone().detach()]
        len_params = theta_i.shape[-1]
        optimizer_torch = torch.optim.SGD([theta_i], lr=0.5,weight_decay=1e-5)

        
        self.__dict__.update(locals())
        
        self.trainer = None ## initialized by trainer
        self.n_samples = 0
        self.n_grad_steps = 0
        
        print(f' Gibo will use {self.n_max} last points to fit GP and {self.n_info_samples} info samples')
        
    
    def sample_acqf(self,bounds):
        
        # Optimize acquistion function and get new observation.
        new_x, acq_value = botorch.optim.optimize_acqf(
            acq_function=self.gradInfo,
            bounds=bounds,
            q=1,  # Analytic acquisition function.
            num_restarts=10,
            raw_samples=128,
            options={'nonnegative': True, 'batch_limit': 5},
            return_best_only=True,
            sequential=False)
        
        return new_x,acq_value
        
    def optimize_information(self,objective_env,model,bounds):
        
        
        acq_value_old = None
        n_info = 0
        i = 0
        ## Optimize gradient information locally
        for i in range(self.n_info_samples):
            
            self.n_samples += 1
            n_info += 1

            model.posterior(self.theta_i)  ## hotfix
            new_x,acq_value = self.sample_acqf(bounds)
            new_y,new_s,new_transitions = objective_env(new_x,1)
            
            ### log the real return (not noisy)
            j = new_y
            #j,_,_ = objective_env(new_x,5)
            self.trainer.log(self.n_samples,{"policy_return":j})
            ##############################
            model.append_train_data(new_x,new_y, strict=False) 
            model.append_module_data(new_y,new_s,new_transitions)
            model.posterior(self.theta_i)  ## hotfix
            self.gradInfo.update_K_xX_dx()

            if acq_value_old is not None:
                
                #if (acq_value-acq_value_old) < 1e-2 and n_info > 2:
                if (acq_value-acq_value_old) < 1e-2:
                    
                    break                
            
                #self.trainer.log(self.n_samples,{"acq_diff":acq_value-acq_value_old})
            
            
            #self.log_sample_info(new_x)
            acq_value_old = acq_value
        
        #self.trainer.log(self.n_samples,{"acq_value (after finish)":acq_value})
        #self.trainer.log(self.n_samples,{"n_info_points":n_info})
                              
    def log_sample_info(self,new_x):
        
        theta_i = self.theta_i
        
        kernel_states = self.model.covar_module.states
        a1 = self.model.covar_module.run_parameters(new_x,kernel_states).squeeze().T
        a2 = self.model.covar_module.run_parameters(theta_i,kernel_states).squeeze().T
        
        param_distance_to_local = torch.linalg.norm(theta_i-new_x)
        action_distance_to_local = torch.linalg.norm(a1-a2)
        
        self.trainer.log(self.n_samples,{
            "param_distance_to_local":param_distance_to_local,
            "action_distance_to_local":action_distance_to_local,
        })



    def current_actions(self,params):
        
            kernel_states = self.model.covar_module.states
            a = self.model.covar_module.mlp(kernel_states,params).flatten()
            return a
        
    def compute_inv_hessian(self,params):
        
        kernel_states = self.model.covar_module.states    
        n_ard_dims = kernel_states.shape[0]
        lengthscales = self.model.covar_module.base_kernel.lengthscale.squeeze()
        
            
        
        grads = torch.autograd.functional.jacobian(self.current_actions,params).squeeze()
        grads = ( (1/lengthscales) * grads.T).T
        hessian = (1/n_ard_dims)*(grads.T @ grads)
        n = hessian.shape[0]
        inv_hessian = torch.linalg.inv(hessian + 1e-1*torch.eye(n))
        
        return inv_hessian

        
    def one_gradient_step(self,model,theta_i):
        
        self.n_grad_steps +=1            
    
        inv_hessian = None
        
        ## compute gradients manually
        with torch.no_grad():
          
            self.optimizer_torch.zero_grad()
            mean_d, variance_d = model.posterior_derivative(theta_i)
            params_grad = mean_d.view(1, self.len_params)
            
            if self.normalize_gradient:
                
                # lengthscale = model.covar_module.base_kernel.lengthscale.detach()
                # params_grad = torch.nn.functional.normalize(params_grad) * lengthscale
                
                params_grad = torch.nn.functional.normalize(params_grad)
                inv_hessian = self.compute_inv_hessian(self.theta_i)
                params_grad = (inv_hessian @ params_grad.T).T     
            
            # if self.standard_deviation_scaling:
            #     params_grad = params_grad / torch.diag(variance_d.view(self.len_params, self.len_params))

            theta_i.grad = -params_grad  # set gradients
            self.optimizer_torch.step()  
            #self.log_grads(mean_d,variance_d,params_grad,inv_hessian) 
            #self.model.log_hypers(self.n_samples)
            
    
    def fit_model_hypers(self,model):
        
        if (model.N >= self.n_max): 
            
            # Restrict data to only recent points
            last_x = model.train_inputs[0][-self.n_max:]
            last_y = model.train_targets[-self.n_max:]
            model.set_train_data(last_x,last_y,strict=False)
            model.posterior(self.theta_i)  ## hotfix
            
            # Adjust hyperparameters
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)
            
        
    def step(self,model,objective_env):
        
        
        self.n_samples += self.n_eval
        
        # Theta_i is directly updated by gradient
        theta_i = self.theta_i
    
        # Evaluate current parameters
        local_y,local_s,local_transitions = objective_env(theta_i,self.n_eval)
        
        ### log the real policy return not noisy
        j = local_y
        #j,_,_ = objective_env(theta_i,5)            
        self.trainer.log(self.n_samples,{"policy_return":j,"policy_return_at_grad":j})
        ###########################################
        
        model.append_train_data(theta_i,local_y, strict=False)
        model.set_module_data(local_y,local_s,local_transitions)
        
        targets = self.trainer.model.y_hist.squeeze().numpy()
        self.trainer.log(self.n_samples,{"max_return":targets.max()})
        
        # Sample locally to optimize gradient information
        self.gradInfo.update_theta_i(theta_i) ## this also update KxX_dx
        bounds = torch.tensor([[-self.delta], [self.delta]]) + theta_i
        self.optimize_information(objective_env,model,bounds)
        
        ## NEEEW : Adjust hyperparameters after information collection for better gradient estimate
        self.fit_model_hypers(model)
     
        # Take one step in direction of the gradient
        self.one_gradient_step(model, theta_i)

        # Add new theta_i to history 
        self.params_history.append(theta_i.clone().detach())
        
        
        #print(f'agent actor params {self.agent._actor_approximator.model.network.default_weights.data}')

    def log_grads(self,mean_d,variance_d,params_grad,inv_hessian=None):
        
        mean_d = torch.abs(mean_d)
        variance_d = torch.abs(torch.diag(variance_d.squeeze()))
        params_grad = torch.abs(params_grad)
        variance_mean = variance_d / mean_d
        
        dct = {
            "max_grad(before hessian)":mean_d.max(),
            "mean_grad(before hessian)":mean_d.mean(),
            "max_covar(before hessian)":variance_d.max(),
            "max_var/mean": variance_mean.max(), 
            }
        
    
        if inv_hessian is not None:
            
            inv_hessian_eigvals = torch.linalg.eigh(inv_hessian)[0]
            dct.update({
                "inv_hessian_avg_eigvalue":torch.mean(inv_hessian_eigvals),
                "max_grad(after hessian)":params_grad.max(),
                "mean_grad(after hessian)":params_grad.mean(),
            })
            
        self.trainer.log(self.n_samples,dct)
        
        
