import fractions
import logging
import logging.config

import botorch
import gpytorch
from numpy import var
import torch
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from src.gp.kernels import *  # # delte later
from src.optimizers.helpers import sparsify
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
            thetas: A (q) x D-dim Tensor of (q) batches with a d-dim theta points each.

        Returns:
            A (q)-dim Tensor of acquisition function values at the given theta points.
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
                normalize_gradient,confidence_gradient,adaptive_lr,
                delta,learning_rate):

        gradInfo = GradientInformation(model)
        theta_i = agent._actor_approximator.model.network.default_weights.data
        params_history = [theta_i.clone().detach()]
        len_params = theta_i.shape[-1]
        optimizer_torch = torch.optim.SGD([theta_i], lr=learning_rate)

        
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
            num_restarts=self.n_info_samples*5,
            raw_samples=128,
            options={'nonnegative': True, 'batch_limit': 5,"seed":1},
            return_best_only=True,
            sequential=False)
        
        return new_x,acq_value
        
    def optimize_information(self,objective_env,model,bounds):
        
        
        acq_value_old = None
        n_info = 0
        i = 0
        ## Optimize gradient information locally
        for i in range(self.n_info_samples):
            
            self.n_samples += self.n_eval
            n_info += 1

            model.posterior(self.theta_i)  ## hotfix
            new_x,acq_value = self.sample_acqf(bounds)
            new_y,new_s,new_transitions,var_reward = objective_env(new_x,self.n_eval)
            
            ### log the real return (not noisy)
            j = new_y
            #j,_,_ = objective_env(new_x,5)
            self.trainer.log(self.n_samples,{"policy_return":j,"episode_length":new_s.shape[0]})
            ##############################
            model.append_train_data(new_x,new_y, strict=False) 
            model.append_module_data(new_y,new_s,new_transitions)
            model.posterior(self.theta_i)  ## hotfix
            self.gradInfo.update_K_xX_dx()

            # if acq_value_old is not None:
                
            #     if (acq_value-acq_value_old) < 1e-2:
                    
            #         break                
            
                #self.trainer.log(self.n_samples,{"acq_diff":acq_value-acq_value_old})
            
            
            self.log_sample_info(objective_env,self.theta_i,new_x,is_grad=False)
            acq_value_old = acq_value
        
        
        self.trainer.log(self.n_samples,{
            "n_info_points":n_info,
            "acq_value (after finish)":acq_value,
            
            })


    def compute_inv_hessian(self,params):
        
        kernel_states = self.model.covar_module.states    
        n_states = kernel_states.shape[0]
        lengthscales = self.model.covar_module.base_kernel.lengthscale.squeeze()
        
        normalized_actions = lambda params : self.current_actions(params,lengthscales)
        
        grads = torch.autograd.functional.jacobian(normalized_actions,params)

        grads = grads.squeeze()

        #grads = ( (1/lengthscales) * grads.T).T
        hessian = (1/n_states)*(grads.T @ grads)
        n = hessian.shape[0]
        hessian = hessian + 1e-3*torch.eye(n)

        ### Replace 0 entries on the diagonal
        diag = torch.diag(hessian)
        diag[diag==0]=1e-3
        hessian.diagonal().copy_(diag)
        #####
        inv_hessian = torch.linalg.inv(hessian)
        
        return inv_hessian

        
    def one_gradient_step(self,model,theta_i):
        
        self.n_grad_steps +=1            
    
        inv_hessian = None
        
        ## compute gradients manually
        with torch.no_grad():
          
            self.optimizer_torch.zero_grad()
            mean_d, variance_d = model.posterior_derivative(theta_i)
            params_grad = mean_d.view(1, self.len_params)


            if self.confidence_gradient :

                #variance_d = torch.diag(variance_d.squeeze())
                params_grad,fraction = sparsify(params_grad,variance_d)
                self.trainer.log(self.n_samples,{"fraction_sparse":fraction})

            if self.normalize_gradient:
                
                if isinstance(self.model.covar_module,StateKernel):

                    
                    inv_hessian = self.compute_inv_hessian(self.theta_i)
                    params_grad = (inv_hessian @ params_grad.T).T   
                    params_grad = torch.nn.functional.normalize(params_grad)

                else :  

                    params_grad = torch.nn.functional.normalize(params_grad)
                    lengthscale = model.covar_module.base_kernel.lengthscale.detach()
                    params_grad = lengthscale * params_grad
            
            
            if self.adaptive_lr :

                 r_variance = model.train_targets.var()
                 params_grad = params_grad/r_variance
                
            
            theta_i.grad = -params_grad  # set gradients
            self.optimizer_torch.step()  
            self.log_grads(mean_d,variance_d,params_grad,inv_hessian) 
            self.model.log_hypers(self.n_samples)
    
        
    def step(self,model,objective_env):
        
        
        self.n_samples += 2*self.n_eval
        
        # Theta_i is directly updated by gradient
        theta_i = self.theta_i
        old_theta_i = self.theta_i.clone().detach()
    
        # Evaluate current parameters
        local_y,local_s,local_transitions,var_reward = objective_env(theta_i,2*self.n_eval)
        
        ### log the real policy return not noisy
        j = local_y
        #j,_,_ = objective_env(theta_i,5)            
        self.trainer.log(self.n_samples,{"policy_return":j,"policy_return_at_grad":j,"episode_length":local_s.shape[0],"var_reward":var_reward})
        ###########################################
        
        model.append_train_data(theta_i,local_y, strict=False)
        model.set_module_data(local_y,local_s,local_transitions)
        
        targets = self.trainer.model.y_hist.squeeze().numpy()
        self.trainer.log(self.n_samples,{"max_return":targets.max()})

        #Adjust hyperparameters before information collection
        self.fit_model_hypers(model)
     
        # Sample locally to optimize gradient information
        self.gradInfo.update_theta_i(theta_i) ## this also update KxX_dx
        bounds = torch.tensor([[-self.delta], [self.delta]]) + theta_i
        self.optimize_information(objective_env,model,bounds)
     
        # Take one step in direction of the gradient
        self.one_gradient_step(model, theta_i)

        # Add new theta_i to history 
        self.params_history.append(theta_i.clone().detach())

        # Compute distance between successive gradient points

        self.log_sample_info(objective_env,self.theta_i,old_theta_i,is_grad=True)
        
        
    
    def current_actions(self,params,lengthscales):
        
            kernel_states = self.model.covar_module.states
            a = self.model.covar_module.mlp(kernel_states,params)#.flatten()
            a = a/ lengthscales
            a = a.flatten()

            return a

    def fit_model_hypers(self,model):
        
        if (model.N >= self.n_max): 
            
            # Restrict data to only recent points
            last_x = model.train_inputs[0][-self.n_max:]
            last_y = model.train_targets[-self.n_max:]
            model.set_train_data(last_x,last_y,strict=False)
            model.posterior(self.theta_i)  ## hotfix
            
            # Adjust hyperparameters
            mll = ExactMarginalLogLikelihood(model.likelihood, model) ## max_retries = 20/default max_retries is 5
            fit_gpytorch_model(mll)
            

    def log_grads(self,mean_d,variance_d,params_grad,inv_hessian=None):
        
        variance_d = variance_d.squeeze()
        mean_d = torch.abs(mean_d)
        std_d = torch.sqrt(torch.abs(torch.diag(variance_d)))
        params_grad = torch.abs(params_grad)
        variance_mean = std_d / mean_d
        
        non_diag = variance_d - torch.diag(variance_d)

        dct = {
            "max_grad(before hessian)":mean_d.max(),
            "mean_grad(before hessian)":mean_d.mean(),
            "median_grad(before hessian)":mean_d.median(),
            "max_covar(before hessian)":variance_d.max(),
            "max_std/mean": variance_mean.max(), 
            "mean_std/mean": variance_mean.mean(), 
            "median_std/mean": variance_mean.median(),
            "max_covar" :non_diag.max(),
            "median_covar" :non_diag.median(),
            "mean_covar" :non_diag.mean(),
            }
        
        if inv_hessian is not None:
            
            inv_hessian_eigvals = torch.linalg.eigh(inv_hessian)[0]
            dct.update({
                "inv_hessian_avg_eigvalue":torch.mean(inv_hessian_eigvals),
                "inv_hessian_median_eigvalue":torch.median(inv_hessian_eigvals),
                "max_grad(after hessian)":params_grad.max(),
                "mean_grad(after hessian)":params_grad.mean(),
                "median_grad(after hessian)":params_grad.median(),
            })
            
        self.trainer.log(self.n_samples,dct)


    
    def log_sample_info(self,objective_env,theta_1,theta_2,is_grad=False):
        
        mlp = objective_env.mlp
        kernel_states = self.model.covar_module.states

        a1 = mlp(kernel_states,theta_1).flatten(start_dim=-2).squeeze().T
        a2 = mlp(kernel_states,theta_2).flatten(start_dim=-2).squeeze().T

        
        param_distance_to_local = torch.abs(theta_1-theta_2).mean()
        n = a1.shape[0]
        action_distance_to_local = torch.abs(a1.double()-a2.double()).mean()
        

        if is_grad : 

            self.trainer.log(self.n_samples,{
                        "param_distance_grad":param_distance_to_local,
                        "action_distance_grad":action_distance_to_local,
                        })


        else : 
        
            self.trainer.log(self.n_samples,{
                "param_distance_to_local":param_distance_to_local,
                "action_distance_to_local":action_distance_to_local,
            })

        

        
        
