import logging
import logging.config

import torch
### botorch
from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

logging.config.fileConfig('logging.conf')
logger = logging.getLogger("MathLog."+__name__)


class BOptimizer(object):
          
  def step(self,model,objective_env):
        
      len_params = model.train_inputs[0].shape[-1]
      
      ### fit hypers of GP
      mll = ExactMarginalLogLikelihood(model.likelihood, model)
      fit_gpytorch_model(mll)

      ### optimize acqf
      best_value = model.train_targets.max()
      EI = ExpectedImprovement(model=model, best_f=best_value)
      eps = 1e-2
      new_x, _ = optimize_acqf(
        acq_function=EI,
        bounds = torch.tensor([[-eps] * len_params, [eps] * len_params]),
        q=1, ## always 1 for closed form acqf
        raw_samples=20, ##number of initial random samples  
        num_restarts=5, ## number of seeds initiated from random restarts
      )
      
      ### evaluate new_x (here we evaluate only once)
      new_y,new_s = objective_env(new_x)
      
      assert not new_x.requires_grad    
      ### Update training points.
      model.update_train_data(new_x,new_y,new_s, strict=False)
      
      return new_x,new_y,new_s
  
    
  def print_hypers(self,model):
        
    #   print("##############################")
    #   for name,param in model.named_parameters():
    #       if param.requires_grad:
    #         print(name, param.data)
      print("##############################")
      print(f'covar_lengthscale {model.covar_module.kernel.base_kernel.lengthscale} \
            covar_outputscale {model.covar_module.kernel.outputscale.item()} \
            noise {model.likelihood.noise_covar.noise.item()}')
      print("##############################")
            
    
def my_optimize_hyps():
    
    training_iter = 100 
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(3):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        logger.warning(f'Loss {loss.shape}')
        loss.backward()
        print('Iter %d/%d - Loss: %.3f noise: %.3f' % 
            (
            i + 1, training_iter, loss.item(),
            model.likelihood.noise.item())
            )
        optimizer.step()

def my_optimize_acqf(acq_function,bounds,
                     q,num_restarts,raw_samples):
  
  N,r = num_restarts,raw_samples
  d = bounds.shape[-1]
  
  # generate a large number of random q-batches
  # these will be used to generate promising samples
  Xraw = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(r, q, d)
  Yraw = acq_function(Xraw)  # evaluate the acquisition function on these q-batches

  # apply the heuristic for sampling promising initial conditions
  X = initialize_q_batch_nonneg(Xraw, Yraw, N)

  # we'll want gradients for the input
  X.requires_grad_(True)
  
  # set up the optimizer, make sure to only pass in the candidate set here
  optimizer = torch.optim.Adam([X], lr=0.01)
  old_loss,new_loss = torch.Tensor([-1e-1]),torch.Tensor([-1])

  # run a basic optimization loop
  for i in range(75):
      optimizer.zero_grad()
      # this performs batch evaluation, so this is an N-dim tensor
      losses = - acq_function(X)  # torch.optim minimizes
      loss = losses.sum()
      
      loss.backward()  # perform backward pass
      optimizer.step()  # take a step
      
      # clamp values to the feasible set
      for j, (lb, ub) in enumerate(zip(*bounds)):
          X.data[..., j].clamp_(lb, ub) # need to do this on the data not X itself
      
        
      #   if (i + 1) % 15 == 0:
      #       print(f"Iteration {i+1:>3}/75 - Loss: {loss.item():>4.3f}")
        
      # use your favorite convergence criterion here...
      crit = (old_loss-new_loss)/old_loss    
      if abs(crit) < 1e-2 : break
  
  X_best = X[torch.argmax(acq_function(X))]
  return X_best.detach(),None

