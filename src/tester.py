import pickle
import random

import botorch
import gpytorch
import matplotlib.pyplot as plt
import torch
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split

from src.kernels import StateKernel

##fix random seed
random.seed(0)
torch.manual_seed(0)

class Tester:
    
    def __init__(self,model,objective_env,
                 local_opt,use_opt_states,delta,
                 n_train,n_test,n_episodes
                 ):
        
        #### if local opt is a path
        if type(local_opt) == str:
            with open(local_opt,'rb') as f:
                local_opt,local_y = pickle.load(f)
                print(f'local_y {local_y} local_opt {local_opt[:10]}')
        
        self.__dict__.update(locals())
        self.test_optimum(local_opt)
        
        
    def test_optimum(self,local_opt):
        
        rewards = []
        for i in range(10):
            
            reward,states = self.objective_env.run(local_opt)
            rewards.append(reward.item())
            
        print(f' intial local opt reward : {sum(rewards)/len(rewards)}')
                
    def run_params(self,x,n_episodes):
            
        tmp = [self.objective_env(p,n_episodes) for p in x]
        y = torch.Tensor([d[0] for d in tmp]).reshape(-1)  ## [n_trials,1]
        s = torch.stack( [d[1] for d in tmp])  ## [n_trials,max_len,state_dim]
        s = torch.flatten(s,start_dim=0,end_dim=1) ## [n_trials*max_len,state_dim]
        
        return (x,y,s)
       
    def generate_data(self):
        
        local_opt,delta = self.local_opt,self.delta
        n_train,n_test = self.n_train,self.n_test
        
        print(f'Generating data')
        
        ### sample parameters unformly
        bounds = torch.tensor([[-delta], [delta]]) + local_opt
        U = torch.distributions.Uniform(bounds[0],bounds[1])
        data_x = U.sample(sample_shape=[n_train+n_test])
        
        ### split data
        train_x,test_x = train_test_split(data_x,test_size=n_test)
        train_data = self.run_params(train_x,n_episodes=1) ## run train points only once
        test_data = self.run_params(test_x,self.n_episodes) ## run test points multiple times to get real value
        _,opt_states = self.objective_env(local_opt.reshape(1,-1))
        
        print(f'Done generating data')
        
        return train_data,test_data,opt_states
    
  
  
    def plot(self,x,y,y_hat,best_x,title,
             subfigure=None):
    
        
        ### arrange points by l2 distance to optimum
        dist = torch.linalg.norm(x-best_x,dim=1)
        idx = torch.argsort(dist)
 
        y = y[idx]
        y_hat = y_hat[idx]
        x_plot = range(len(y))
        
        if subfigure is None:
            fig,axs = plt.subplots(3,figsize=(5,12))

        else :
            axs = subfigure.subplots(3)
            
        axs[0].scatter(x_plot,y,label="y",color="red")
        axs[0].errorbar(x_plot,y_hat,label="y_hat",fmt="x")
        axs[0].title.set_text(f' Predictions vs targets \n MAE: {mae(y,y_hat)} \n RMSE :{mse(y,y_hat,squared=False)}')
        axs[0].set_xlabel("rank (in terms of distance to optimum)")
        axs[0].set_ylabel("prediction")
        
        axs[0].legend()
        
        idx2 = torch.argsort(y)
        axs[1].scatter(y[idx2],y_hat[idx2],label="error")
        axs[1].plot(y[idx2],y[idx2],color="red")
        axs[1].title.set_text("Error as a function of the target")
        axs[1].set_xlabel("y")
        axs[1].set_xlabel("y_hat")
        
        error = torch.abs(y[idx]-y_hat[idx])
        axs[2].scatter(dist[idx],torch.cumsum(error,dim=0))
        axs[2].set_title("Cummulative error as a function of L2 distance to optimum")
        axs[2].set_xlabel("||x - x_opt||")
        axs[2].set_xlabel("Cummulative error")
        
        plt.legend()
        
        
    
    def get_mll(self,model,x,y):
        
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        output = model(x)
        marginal_likelihood = mll(output,y)
        
        return marginal_likelihood.item()
        
    def predict(self,model,x):
        
        model.eval()
        model.likelihood.eval()
        
        with torch.no_grad():

            pred = model.likelihood(model(x))
        
        lower, upper = pred.confidence_region()   
        y_hat = pred.mean  
        
        return y_hat,lower,upper
    
    def fit(self,model,train_x,train_y,
            opt_states,use_opt_states):
        
        model.set_train_data(train_x,train_y,opt_states,strict=False)
        model.train()
        model.likelihood.train()
        
        ### Check hypers before training
        model.print_hypers()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        
        ### Check hypers after training
        model.print_hypers()
        
        return model


    def run(self,train_data,test_data,opt_states):
        
        self.__dict__.update(locals()) ##save args to self
        
         
        # fit model locally
        train_x,train_y,train_s = train_data
        model = self.fit(self.model,train_x,train_y,opt_states,self.use_opt_states)

        # generate predictions for test observations
        
        test_x,test_y,test_s = test_data

        train_pred = self.predict(model,train_x)
        test_pred  = self.predict(model,test_x)
        
        train_mll = self.get_mll(model,train_x,train_y)
        test_mll = self.get_mll(model,test_x,test_y)
        
        
        #fig = plt.figure(constrained_layout=True,figsize=(8,12))
        fig = plt.figure(constrained_layout=True,figsize=(10,12))
        subfigures = fig.subfigures(1,2)
        ### train plots
        x,y,_ = train_data
        y_hat,lower,upper = train_pred
        self.plot(x,y,y_hat,self.local_opt,"train",
                  subfigures[0])
        
        ### test plots
        x,y,_ = test_data
        y_hat,lower,upper = test_pred
        self.plot(x,y,y_hat,self.local_opt,"test",
                  subfigures[1])
        
        opt_data = (self.local_opt,opt_states)

        
        return train_data,train_pred,test_data,test_pred,opt_data
    
    
    

