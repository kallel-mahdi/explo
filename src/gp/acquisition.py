import torch
import botorch

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
    @botorch.utils.transforms.t_batch_mode_transform()
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




class StateGradientInformation(botorch.acquisition.AnalyticAcquisitionFunction):
    '''Stochastic version of GradientInformation
    Allows to compute information using a subset of state for faster computations.
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
        
        # original gibo uses this to cash part of the computation
        pass
    
    def K_xX(self,theta_t,X_hat,kernel):
            
        rslt = kernel(theta_t,X_hat).evaluate()
        
        return rslt
    
    def _get_KθX_dθ(self, theta_t, X_hat,kernel) :
        
        '''Computes the analytic derivative of the kernel K(x,X) w.r.t. x.

        Args:
            x: (n x D) Test points.

        Returns:
            (n x D) The derivative of K(x,X) w.r.t. x.
        '''
        f  = lambda theta : self.K_xX(theta,X_hat,kernel)
        jacobs = torch.autograd.functional.jacobian(func=f,inputs=(theta_t))
        KθX_dθ = jacobs.sum(dim=2).transpose(1,2)

        return KθX_dθ


    @botorch.utils.transforms.t_batch_mode_transform()
    def forward(self, thetas) :
        
        '''Evaluate the acquisition function on the candidate set thetas.

        Args:
            thetas: A (q) x D-dim Tensor of (q) batches with a d-dim theta points each.

        Returns:
            A (q)-dim Tensor of acquisition function values at the given theta points.
        '''
        
        
        kernel = self.model.covar_module.get_mini_kernel(batch_size=100)
        sigma_n = self.model.likelihood.noise_covar.noise
        D = self.model.D
        X = self.model.train_inputs[0] 
        x = self.theta_i.view(-1, D)
        variances = []
        
        for theta in thetas:
            
            theta = theta.view(-1, D)

            # Compute everything from 0
            # Unlike original which cashes half of the computation
        
            X_hat = torch.cat([X,theta])
            K_XX = kernel(X_hat,X_hat).evaluate() + sigma_n * torch.eye(X_hat.shape[0])
            K_XX_inv = torch.linalg.inv(K_XX)            
            K_xX_dx = self._get_KθX_dθ(x, X_hat,kernel)                   
            
            # Compute_variance.
            variance_d = -K_xX_dx @ K_XX_inv @ K_xX_dx.transpose(1, 2)
            variance_d = variance_d.squeeze()
            variances.append(torch.trace(variance_d).view(1))

        return -torch.cat(variances, dim=0)

