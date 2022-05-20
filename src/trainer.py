
from src.optim import BOptimizer

class Trainer:
    
    def __init__(self,model,objective_env,optimizer,
                 n_steps,report_freq):
        
        #optimizer = BOptimizer()
        self.__dict__.update(locals())
    
    def print_hypers(self,model):
            
        #   print("##############################")
        #   for name,param in model.named_parameters():
        #       if param.requires_grad:
        #         print(name, param.data)
        
        print("##############################")
        print(f'covar_lengthscale {model.covar_module.base_kernel.lengthscale} \
                covar_outputscale {model.covar_module.outputscale.item()} \
                noise {model.likelihood.noise_covar.noise.item()}')
        print("##############################")
                
        
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
                self.print_hypers(model)
    
        self.best_x,self.best_y = model.get_best_params()

        return self.best_x,self.best_y
                

