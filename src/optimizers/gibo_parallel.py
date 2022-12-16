import fractions
import logging
import logging.config

import botorch
import gpytorch
from numpy import var
import torch
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from src.gp.kernels import *  
from src.gp.acquisition import *
#from torch.optim import LBFGS ## Full pytorch LBFGS implementation

logging.config.fileConfig('logging.conf')
logger = logging.getLogger("ShapeLog."+__name__)

class GIBOptimizer(object):
        
    def __init__(self,init_params,model,
                n_eval,n_max,n_info_samples,
                normalize_gradient,
                delta,learning_rate):

        
        if isinstance(model.covar_module,StateKernel):
            gradInfo = StateGradientInformation(model)
        else : 
            gradInfo = GradientInformation(model)
        
        theta_i = init_params
        params_history = [theta_i.clone().detach()]
        len_params = theta_i.shape[-1]
        optimizer_torch = torch.optim.SGD([theta_i], lr=learning_rate)

        
        self.__dict__.update(locals())
        
        self.trainer = None ## initialized by trainer
        self.n_samples = 0
        self.n_grad_steps = 0
        self.sparse_counter = 0
        self.max_sparse = self.n_max / self.n_info_samples
        
        print(f' Gibo will use {self.n_max} last points to fit GP and {self.n_info_samples} info samples')
    
    def sample_acqf(self,bounds):
        
        # Optimize acquistion function and get new observation.
        new_x, acq_value = botorch.optim.optimize_acqf(
            acq_function=self.gradInfo,
            bounds=bounds,
            q=self.n_info_samples,  # Analytic acquisition function.
            num_restarts=self.n_info_samples,
            raw_samples=256,
            options={'nonnegative': True, 'batch_limit': 5,"seed":1},
            return_best_only=True,
            sequential=False)
        
        return new_x,acq_value
        
    def optimize_information(self,objective_env,model,bounds):
        
        model.posterior(self.theta_i)  ## hotfix
        acq_pts,acq_value = self.sample_acqf(bounds)

        ##############################
        for new_x in acq_pts:

            self.n_samples += self.n_eval                
            new_x = new_x.reshape(1,-1)
            new_y,new_s,new_transitions,var_reward = objective_env(new_x,self.n_eval)
            model.append_train_data(new_x,new_y, strict=False) 
            model.append_module_data(new_y,new_s,new_transitions)
            ##############################
            ### log the real return (not noisy)
            j = new_y
            #j,_,_ = objective_env(new_x,5)
            self.trainer.log(self.n_samples,{"policy_return":j,"episode_length":new_s.shape[0]})
            self.log_sample_info(objective_env,self.theta_i,new_x,is_grad=False)

        ##############################
        model.posterior(self.theta_i)  ## hotfix
        self.gradInfo.update_K_xX_dx()



    def compute_inv_hessian(self,params):
        
        kernel_states = self.model.covar_module.states    
        n_states = kernel_states.shape[0]
        lengthscales = self.model.covar_module.base_kernel.lengthscale.squeeze()
        
        normalized_actions = lambda params : self.current_actions(params,lengthscales)
        
        grads = torch.autograd.functional.jacobian(normalized_actions,params)

        grads = grads.squeeze()

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


            if self.normalize_gradient:
                
                if isinstance(self.model.covar_module,StateKernel):

                    
                    inv_hessian = self.compute_inv_hessian(self.theta_i)
                    params_grad = (inv_hessian @ params_grad.T).T   
                    params_grad = torch.nn.functional.normalize(params_grad)

                else :  

                    params_grad = torch.nn.functional.normalize(params_grad)
                    lengthscale = model.covar_module.base_kernel.lengthscale.detach()
                    params_grad = lengthscale * params_grad
            
            ### New baseline (makes more sense)
            else : 

                params_grad = torch.nn.functional.normalize(params_grad)
     
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
