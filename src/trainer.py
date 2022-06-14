
import pickle
import matplotlib.pyplot as plt
import  torch
import  numpy as np
class Trainer:
    
    def __init__(self,model,objective_env,optimizer,
                 n_steps,report_freq,save_best=False,):
        
        self.__dict__.update(locals())
    
    
    def save_bests(self):

        task_name = self.objective_env.env.spec.id
        ckpt_path = "/home/q123/Desktop/explo/local_optima/"+task_name+"_"+str(self.model.mlp.Ls)
        
        
        with open(ckpt_path,'wb') as handle:
            
    
            pickle.dump((self.best_x,self.best_y),handle)
            
        print(f'Saved best weights to {ckpt_path}')
        
    
    def plot_cummulative_regret(self):
        
        model = self.model    
        targets = model.y_hist.squeeze().numpy()
        n_trials = targets.shape[0]
        best_performance=np.zeros(n_trials)



        for i in range(1,n_trials):

            best_performance[i] = targets[:i].max()

        plt.plot(best_performance)
                

    def run(self):
        
        report_freq = self.report_freq
        optimizer,model = self.optimizer,self.model
        objective_env = self.objective_env  
        
        for i in range(self.n_steps):
            
            optimizer.step(model,objective_env)
            
            if (i % report_freq) == 0 and i>=report_freq:

                max = model.y_hist.max()
                curr = model.y_hist[-1]
                
                last_batch= model.train_targets[-report_freq:]
                batch_mean = last_batch.mean()
                batch_max  = last_batch.max()
                
                print(f'current {curr} / max {max} /batch_mean {batch_mean} /batch_max {batch_max} ')
                model.print_hypers()
                model.print_train_mll()
    
        self.best_x,self.best_y = model.get_best_params()
        
        self.plot_cummulative_regret()
        
        if self.save_best : self.save_bests()
        

        return self.best_x,self.best_y
                




