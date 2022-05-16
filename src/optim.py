import torch

from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.optim.initializers import initialize_q_batch_nonneg

import logging

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

def step(model,objective_env):
      
    ### see evolution of parameters
    print("##############################")
    for name,param in model.named_parameters():
        if param.requires_grad:
          print(name, param.data)
    
    print("##############################")
    ###########################
    
    len_params = model.train_inputs[0].shape[-1]
    ### fit hypers of GP
    
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    ### optimize acqf
    best_value = model.train_targets.max()
    EI = ExpectedImprovement(model=model, best_f=best_value)
    new_x, _ = optimize_acqf(
      acq_function=EI,
      bounds = torch.tensor([[-1.0] * len_params, [1.0] * len_params]),
      q=1, ## always 1 for closed form acqf
      num_restarts=5,   
      raw_samples=20, ##number of initial random samples  
    )
    
    ### evaluate new_x (here we evaluate only once)
    new_y,new_s = objective_env(new_x)
    
    ### Update training points.
    model.update_train_data(new_x,new_y,new_s, strict=False)
    
    
    return new_x,new_y,new_s