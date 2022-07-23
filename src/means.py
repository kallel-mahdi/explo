import torch
from gpytorch.means import ConstantMean, Mean


class MyConstantMean(ConstantMean):
    
    def __init__(self,*args,**kwargs):
        
        super(MyConstantMean, self).__init__(*args,**kwargs)
    
    
    def set_train_data(self,local_mean,**kwargs):
        
        self.constant.data = local_mean
        

class AdvantageMean(Mean):
    
    def __init__(self,agent):
        
        super(AdvantageMean, self).__init__()
        
        self.register_parameter(name="constant",parameter=torch.nn.Parameter(torch.zeros(1)))
        
        ## initialized later
        self.local_transitions = None
        self.local_params = None
        self.local_states = None
        
    
    def set_train_data(self,local_mean,local_states,local_transitions,local_params):
        
        self.constant.data = local_mean
        self.local_params = local_params
        self.local_states = local_states
        self._replay_memory.add(local_transitions)
        
        
    def update_train_data(self,transitions):
        
        
        self._replay_memory.add(transitions)
    
    
    def fit_critic(self):
        
        self.agent.fit_critic()
            
        
        
    def __call__(self,params):
        
        agent = self.critic_agent 
        local_actions = self.agent._actor_approximator(self.local_params,self.local_states).squeeze().T
        actions = self.agent._actor_approximator(params,self.local_states).squeeze().T
    
        local_q = agent._critic_approximator(self.local_states, local_actions, output_tensor=True, **agent._critic_predict_params)
        q = agent._critic_approximator(self.local_states, actions, output_tensor=True, **agent._critic_predict_params)
        
        mean = self.constant + torch.mean(q-local_q)
        
        return mean
    
    