import botorch
import gpytorch
import torch




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

    def _get_KxX_dx(self, x, X) :
        '''Computes the analytic derivative of the kernel K(x,X) w.r.t. x.

        Args:
            x: (n x D) Test points.

        Returns:
            (n x D) The derivative of K(x,X) w.r.t. x.
        '''
        N = X.shape[0]
        n = x.shape[0]
        K_xX = self.model.covar_module(x, X).evaluate()
        lengthscale = self.model.covar_module.base_kernel.lengthscale.detach()
        return (
            -torch.eye(self.model.D, device=X.device)
            / lengthscale
            @ (
                (x.view(n, 1, self.model.D) - X.view(1, N, self.model.D))
                * K_xX.view(n, N, 1)
            ).transpose(1, 2)
        )

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
            # Compute K_Xθ, K_θθ (do not forget to add noise).
            # K_Xθ = self.model.covar_module(X, theta).evaluate()
            # K_θθ = self.model.covar_module(theta).evaluate() + sigma_n * torch.eye(
            #     K_Xθ.shape[-1]
            # ).to(theta)

            # # Get Cholesky factor.
            # L = one_step_cholesky(
            #     top_left=self.model.get_L_lower().transpose(-1, -2),
            #     K_Xθ=K_Xθ,
            #     K_θθ=K_θθ,
            #     A_inv=self.model.get_KXX_inv(),
            # )

            # # Get K_XX_inv.
            # K_XX_inv = torch.cholesky_inverse(L, upper=True)
            
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


def one_step_cholesky(
    top_left, K_Xθ, K_θθ, A_inv):
    
    '''Update the Cholesky factor when the matrix is extended.

    Note: See thesis appendix A.2 for notation of args and further information.

    Args:
        top_left: Cholesky factor L11 of old matrix A11.
        K_Xθ: Upper right bock matrix A12 of new matrix A.
        K_θθ: Lower right block matrix A22 of new matrix A.
        A_inv: Inverse of old matrix A11.

    Returns:
        New cholesky factor S of new matrix A.
    '''
    # Solve with A \ b: A @ x = b, x = A^(-1) @ b,
    # top_right = L11^T \ A12 = L11^T  \ K_Xθ, top_right = (L11^T)^(-1) @ K_Xθ,
    # Use: (L11^(-1))^T = L11 @ A11^(-1).
    # Hint: could also be solved with torch.cholesky_solve (in theory faster).
    top_right = top_left @ (A_inv @ K_Xθ)
    bot_left = torch.zeros_like(top_right).transpose(-1, -2)
    bot_right = torch.cholesky(
        K_θθ - top_right.transpose(-1, -2) @ top_right, upper=True
    )
    return torch.cat(
        [
            torch.cat([top_left, top_right], dim=-1),
            torch.cat([bot_left, bot_right], dim=-1),
        ],
        dim=-2,
    )