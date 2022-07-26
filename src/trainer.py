
import pickle
import matplotlib.pyplot as plt
import  torch
import  numpy as np
import wandb
class Trainer:
    
    def __init__(self,model,objective_env,optimizer,
                 n_steps,report_freq,
                 save_best=False,
                 wandb_logger=False,
                 run_name=None,
                 wandb_config=None):
        
        self.__dict__.update(locals())
        optimizer.trainer = self
        model.trainer = self
        
        if wandb_logger:
            wandb.init(project="explo",name=run_name,config=wandb_config) 
            
            
    
    def save_bests(self):

        task_name = self.objective_env.env.spec.id
        ckpt_path = "/home/q123/Desktop/explo/local_optima/"+task_name+"_"+str(self.model.mlp.Ls)
        
        
        with open(ckpt_path,'wb') as handle:
            
    
            pickle.dump((self.best_x,self.best_y),handle)
            
        print(f'Saved best weights to {ckpt_path}')
        
 
    def run(self):
        
        report_freq = self.report_freq
        optimizer,model = self.optimizer,self.model
        objective_env = self.objective_env 
        
        
        
        while optimizer.n_samples < self.n_steps :
            
            
            
            optimizer.step(model,objective_env)
            
            
            if (optimizer.n_grad_steps % report_freq) == 0 and optimizer.n_grad_steps>0 :

                max = model.y_hist.max()
                curr = model.y_hist[-1]
                
                last_batch= model.train_targets[-report_freq:]
                batch_mean = last_batch.mean()
                batch_max  = last_batch.max()
                
                print(f'current {curr} / max {max} /batch_mean {batch_mean} /batch_max {batch_max} ')
                model.print_train_mll()
    
        self.best_x,self.best_y = model.get_best_params()
        
        
        if self.save_best : self.save_bests()
        

        return self.best_x,self.best_y

    def log(self,n_samples,dictionary):
        
        
            
        dictionary.update({"n_samples":n_samples})
        
        if self.wandb_logger :
            
            wandb.log(dictionary)
                




