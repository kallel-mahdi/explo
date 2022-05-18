class GIBOptimize(object):
        
               
    def __init__(self,model,
            verbose= False,
            normalize_gradient=True,standard_deviation_scaling=False):

        max_samples_per_iteration = 3
        delta = 0.1
        gi = GradientInformation(model)
        optimizer_torch = torch.optim.SGD(model.parameters(), lr=0.5)
        params_tmp = model.train_inputs[0][-1]
        params_history_list = [params_tmp.clone()]
        D = len_params
        self.__dict__.update(locals())
        
    def optimize_information(self,model,bounds):
    
        ## Optimize gradient information locally
        for i in range(self.max_samples_per_iteration):

            model.posterior(self.curr_params)  ## hotfix

            # Optimize acquistion function and get new observation.
            new_x, _ = botorch.optim.optimize_acqf(
                acq_function=self.gi,
                bounds=bounds,
                q=1,  # Analytic acquisition function.
                num_restarts=20,
                raw_samples=100,
                options={'nonnegative': True, 'batch_limit': 5},
                return_best_only=True,
                sequential=False,
            )
            
            new_y,new_s = objective_env(new_x)
            # Update training points.
            model.update_train_data(new_x,new_y,new_s, strict=False)
            model.posterior(self.curr_params)  ## hotfix
            self.gi.update_K_xX_dx()

    def one_gradient_step():
        
        ## compute gradients manually
        with torch.no_grad():
          
            self.optimizer_torch.zero_grad()
            mean_d, variance_d = model.posterior_derivative(params)
            params_grad = -mean_d.view(1, self.D)

            if self.normalize_gradient:
                lengthscale = model.covar_module.base_kernel.lengthscale.detach()
                params_grad = torch.nn.functional.normalize(params_grad) * lengthscale

            if self.standard_deviation_scaling:
                params_grad = params_grad / torch.diag(variance_d.view(self.D, self.D))

            params.grad = params_grad  # set gradients
            self.optimizer_torch.step()                  

    def step(self,model,objective_env):
        
        
        params = self.params_history_list[-1].reshape(1,-1)
        self.curr_params = params
        model.posterior(self.curr_params)  ## hotfix
 
        # Evaluate current parameters
        f_params = objective_env(params)
        new_y,new_s = objective_env(new_x)
        model.update_train_data(new_x,new_y,new_s, strict=False)
        self.gi.update_theta_i(params)

        # Stay local around current parameters.
        bounds = torch.tensor([[-self.delta], [self.delta]]) + params
        
         # Sample locally to optimize gradient information
        
        self.optimize_information(model,bounds)
        
        # Only optimize model hyperparameters if N >= N_max.
        if (model.N >= model.N_max): 

            # Adjust hyperparameters
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)
            last_x = model.train_inputs[0][-model.N_max:]
            last_y = model.train_targets[-model.N_max:]
            model.set_train_data()
            model.posterior(self.curr_params)  ## hotfix
       