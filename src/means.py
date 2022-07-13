from gpytorch.means import ConstantMean


class MyConstantMean(ConstantMean):
    
    def __init__(self,*args,**kwargs):
        
        super(MyConstantMean, self).__init__(*args,**kwargs)
    
    
    def set_train_data(self,mean,transitions):
        
        self.constant.data = mean
        

class AdvantageMean():
    
    def __init__(self,critic):
        
        pass
    
    
    def set_train_data(self,mean,transitions):
        
        self.constant.data = mean
        self.critic.fit()
    
    def forward(self,x):
        pass
        
        