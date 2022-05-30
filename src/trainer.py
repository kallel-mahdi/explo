
import pickle

from src.optim import BOptimizer


class Trainer:
    
    def __init__(self,model,objective_env,optimizer,
                 n_steps,report_freq,save_best=False,):
        
        #optimizer = BOptimizer()
        self.__dict__.update(locals())
    
    
    def save_bests(self):

        task_name = self.objective_env.env.spec.id
        ckpt_path = "/home/q123/Desktop/explo/local_optima/"+task_name
        
        with open(ckpt_path,'wb') as handle:
    
            pickle.dump((self.best_x,self.best_y),handle)
            
        print(f'Saved best weights to {ckpt_path}')


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
    
        self.best_x,self.best_y = model.get_best_params()
        
        if self.save_best : self.save_bests()

        return self.best_x,self.best_y
                




