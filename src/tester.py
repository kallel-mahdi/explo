import pickle
import random

import botorch
import gpytorch
import matplotlib.pyplot as plt
import torch
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split

##fix random seed
random.seed(0)
torch.manual_seed(0)

class Tester:
    
    def __init__(self,model,objective_env,
                 local_opt,delta,
                 n_train,n_test,n_episodes=5
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
                
                
    def generate_data(self,local_opt,delta,n_train,n_test):
        
        ### sample parameters unformly
        bounds = torch.tensor([[-delta], [delta]]) + local_opt
        U = torch.distributions.Uniform(bounds[0],bounds[1])
        data_x = U.sample(sample_shape=[n_train+n_test])
        
        ### split data
        train_x,test_x = train_test_split(data_x,test_size=n_test)
        train_data = self.run_params(train_x)
        test_data = self.run_params(test_x)
        
        return train_data,test_data
    
  
    def run_params(self,x):
        
        tmp = [self.objective_env(p,self.n_episodes) for p in x]
        y = torch.Tensor([d[0] for d in tmp]).reshape(-1)  ## [n_trials,1]
        s = torch.stack( [d[1] for d in tmp])  ## [n_trials,max_len,state_dim]
        s = torch.flatten(s,start_dim=0,end_dim=1) ## [n_trials*max_len,state_dim]
        
        return (x,y,s)
    
    def plot(self,data,pred_data,mll,best_x,title):
        
        ### get mean and confidence intervals from posterior
        x,y,_ = data
        y_hat,lower,upper = pred_data
        err = lower-y_hat
        
        # print("y_hat",y_hat)
        # print("lower",lower)
        # low_error = (lower-y_hat).reshape(1,-1)
        # up_error  = (upper-y_hat).reshape(1,-1)
        #y_err = torch.vstack((low_error,up_error))
        
        ### arrange points by l2 distance to optimum
        dist = torch.linalg.norm(x-best_x,dim=1)
        idx = torch.argsort(dist)
        y = y[idx]
        y_hat = y_hat[idx]

        ### plot predictions with confidence
        x_plot = range(len(y))
        plt.scatter(x_plot,y,label="true",color="red")
        #plt.errorbar(x_plot,y_hat,yerr = err,label="prediction",fmt="o")
        plt.errorbar(x_plot,y_hat,label="prediction",fmt="x")
        plt.title(title+" MLL: "+str(mll)+'\n'+ "MAE"+str(mae(y,y_hat)))
        plt.legend()
        plt.show()
        
    
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
    
    def fit(self,model,train_x,train_y):
        
        model.train()
        model.likelihood.train()
        
        ### Check hypers before training
        model.print_hypers()
        
        model.set_train_data(inputs=train_x,targets=train_y,strict=False)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        
        ### Check hypers after training
        model.print_hypers()
        
        return model,model(train_x)


    def run(self):
        
        # create data around optimum
        print(f'Generating data')
        train_data,test_data = self.generate_data(
                    self.local_opt,self.delta,
                    self.n_train,self.n_test)
        print(f'Done generating data')
        # fit model locally
        train_x,train_y,train_s = train_data
    
        model,tmp = self.fit(self.model,train_x,train_y)

        # generate predictions for test observations
        
        test_x,test_y,test_s = test_data

        train_pred = self.predict(model,train_x)
        test_pred  = self.predict(model,test_x)
        
        train_mll = self.get_mll(model,train_x,train_y)
        test_mll = self.get_mll(model,test_x,test_y)
        
        self.plot(train_data,train_pred,train_mll,self.local_opt,title="train")
        self.plot(test_data,test_pred,test_mll,self.local_opt,title="test")

        
        return train_data,train_pred,test_data,test_pred,tmp
    
    
    

