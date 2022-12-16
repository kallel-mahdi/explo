import torch
from gpytorch.means import ConstantMean, Mean


class MyConstantMean(ConstantMean):
    
    def __init__(self,*args,**kwargs):
        
        super(MyConstantMean, self).__init__(*args,**kwargs)
        self.constant.requires_grad = True
    
    
    def set_train_data(self,local_mean,*args,**kwargs):
        
        ### Sometimes the learned constant mean is wierd and it's better to be fixed manually
        
        #self.constant.data = local_mean 
        pass
        
    
    def append_train_data(self,new_mean,new_states,new_transitions):
        pass
        
