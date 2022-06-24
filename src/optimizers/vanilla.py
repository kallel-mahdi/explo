import logging
import logging.config

import torch
import gpytorch
### botorch
from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

logging.config.fileConfig('logging.conf')
logger = logging.getLogger("MathLog."+__name__)


class BOptimizer(object):
      
  def __init__(self,n_eval,**kwargs):
        
        self.n_eval = n_eval
          
  def step(self,model,objective_env):
        
      len_params = model.train_inputs[0].shape[-1]
      
      ### fit hypers of GP
      mll = ExactMarginalLogLikelihood(model.likelihood, model)
      fit_gpytorch_model(mll)

      ### optimize acqf
      best_value = model.train_targets.max()
      EI = ExpectedImprovement(model=model, best_f=best_value)
      eps = 1e-1
      new_x, _ = optimize_acqf(
        acq_function=EI,
        bounds = torch.tensor([[-eps] * len_params, [eps] * len_params]),
        q=1, ## always 1 for closed form acqf
        raw_samples=20, ##number of initial random samples  
        num_restarts=5, ## number of seeds initiated from random restarts
      )
      
      ### evaluate new_x (here we evaluate only once)
      new_y,new_s,_ = objective_env(new_x,self.n_eval)
      
      assert not new_x.requires_grad    
      ### Update training points.
      model.append_train_data(new_x,new_y,new_s, strict=False)
      
      return new_x,new_y,new_s
            
    