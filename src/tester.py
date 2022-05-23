
class Tester:
    
    def __init__(self,model,env_objective,
                 local_opt,delta,
                 n_train,n_test
                 ):
        
        
        train_data,test_data = self.generate_data(
            local_opt,delta,n_train,n_test
        )
        
        self.__dict__.update(locals())
        
    
    def generate_data(self,local_opt,n_train,n_test):
        
        bounds = 
        data_x = ## generate local samples
        train_x,test_x = ## split data_x
        train_data = self.run_params(train_x)
        test_data = self.run_params(test_x)
        
        return train_data,test_data
    
    def plot(self):
        
        
         
         
    def run_params(self,x):
        
        tmp = [objective_env.run(p) for p in x]
        y = torch.Tensor([d[0] for d in tmp]).reshape(-1)  ## [n_trials,1]
        s = torch.stack( [d[1] for d in tmp])  ## [n_trials,max_len,state_dim]
        s = torch.flatten(s,start_dim=0,end_dim=1) ## [n_trials*max_len,state_dim]
        
        return (x,y,s)
    

    def run(model):
        
        # fit model
        
        
        # 
            
        with torch.no_grad(), gpytorch.settings.fast_pred_var():

            observed_pred = likelihood(model(test_x))

        
        lower, upper = observed_pred.confidence_region()          
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')            
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
                
            
    
    
   