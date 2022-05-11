import torch

from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf


def step(model,objective_env):
    
    
    ### fit hypers of GP
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    ### optimize acqf
    best_value = model.train_targets.max()
    len_params = objective_env.policy.len_params
    EI = ExpectedImprovement(model=model, best_f=best_value)
    
    new_x, _ = optimize_acqf(
      acq_function=EI,
      bounds=torch.tensor([[0.0] * len_params, [1.0] * len_params]),
      q=1,
      num_restarts=1,
      raw_samples=1,
      options={},
    )

    new_y = objective_env(new_x)
  
    ### Update training points.
    train_x = torch.cat([model.train_inputs[0], new_x])
    train_y = torch.cat([model.train_targets, new_y])
    model.set_train_data(inputs=train_x, targets=train_y, strict=False)