import botorch
import gpytorch
import torch


def my_fit_gpytorch_model(model):
    
    training_iter = 100 
    train_x = model.train_inputs[0]
    train_y = model.train_targets
    # Find optimal model hyperparameters
    model.train()
    model.likelihood.train()

    # Use the adam optimizer

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
    
    print('Likelihood: %.3f noise: %.3f' % 
            (
            - loss.item(),
            model.likelihood.noise.item())
            )

def my_optimize_acqf(acq_function,bounds,
                     q,num_restarts,raw_samples):
  
    N,r = num_restarts,raw_samples
    d = bounds.shape[-1]
    
    # generate a large number of random q-batches
    # these will be used to generate promising samples
    Xraw = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(r, q, d)
    Yraw = acq_function(Xraw)  # evaluate the acquisition function on these q-batches

    # apply the heuristic for sampling promising initial conditions
    X = botorch.optim.initialize_q_batch_nonneg(Xraw, Yraw, N)

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

