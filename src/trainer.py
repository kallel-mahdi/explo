
from src.optim import BOptimizer

class Trainer:
    
    def __init__(self,model,objective_env,optimizer,n_steps):
        
        #optimizer = BOptimizer()
        self.__dict__.update(locals())
        
    def run(self,report_freq=100):
        
        optimizer,model = self.optimizer,self.model
        objective_env = self.objective_env  
        
        for i in range(self.n_steps):
            
            optimizer.step(model,objective_env)
            
            if i % report_freq == 0 and i>=report_freq:

                max = model.train_targets.max()
                batch_mean = model.train_targets[i-report_freq:i].mean()
                batch_max = model.train_targets[i-report_freq:i].max()
                curr = model.train_targets[-report_freq]
                print(f'current {curr} / max {max} /batch_mean {batch_mean} /batch_max {batch_max} ')
                optimizer.print_hypers(model)
    
        self.best_x,self.best_y = model.get_best_params()

        return self.best_x,self.best_y
                

