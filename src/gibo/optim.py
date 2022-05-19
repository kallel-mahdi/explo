import botorch
import torch
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from src.gibo.acqf import GradientInformation

from torch.optim import  LBFGS
class GIBOptimizer(object):
        
               
    def __init__(self,model,
            n_info_samples=16,delta=0.1,
            normalize_gradient=True,standard_deviation_scaling=False,
            verbose= False):

        gradInfo = GradientInformation(model)
        theta_i = model.train_inputs[0][-1].reshape(1,-1)
        params_history = [theta_i.clone()]
        len_params = theta_i.shape[-1]
        N_max = model.N_max ##hotfix
        
        optimizer_torch = torch.optim.SGD([theta_i], lr=0.5)
        
        self.__dict__.update(locals())
        
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
            new_y,new_s = objective_env(new_x)
            model.update_train_data(new_x,new_y,new_s, strict=False)
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
        
        #model.posterior(self.theta_i)  ## hotfix
 
        # Evaluate current parameters
        new_y,new_s = objective_env(theta_i)
        model.update_train_data(theta_i,new_y,new_s, strict=False)
        self.gradInfo.update_theta_i(theta_i)
        
        # Only optimize model hyperparameters if N >= N_max.
        if (model.N >= self.N_max): 
            
            print("TRUNCAAATING OLD DATAAAAAAAAAA")
            self.print_hypers(model)
            

            # Adjust hyperparameters
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)
            # Restrict data to only recent points
            last_x = model.train_inputs[0][-self.N_max:]
            last_y = model.train_targets[-self.N_max:]
            model.set_train_data(inputs=last_x,targets=last_y,strict=False)
            model.N = last_x.shape[0]
            model.posterior(self.theta_i)  ## hotfix
            self.gradInfo.update_K_xX_dx() ## hotfix
        
        # Sample locally to optimize gradient information
        bounds = torch.tensor([[-self.delta], [self.delta]]) + theta_i
        self.optimize_information(objective_env,model,bounds)
        
        # Take one step in direction of the gradient
        self.one_gradient_step(model, theta_i)
        
        # Add new theta_i to history 
        self.params_history.append(theta_i.clone())
        
    def print_hypers(self,model):
        
        #   print("##############################")
        #   for name,param in model.named_parameters():
        #       if param.requires_grad:
        #         print(name, param.data)
        print("##############################")
        print(f'covar_lengthscale {model.covar_module.base_kernel.lengthscale} \
                covar_outputscale {model.covar_module.outputscale.item()} \
                noise {model.likelihood.noise_covar.noise.item()}')
        print("##############################")
                