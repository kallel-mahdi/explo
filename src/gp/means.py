import torch
from gpytorch.means import ConstantMean, Mean


class MyConstantMean(ConstantMean):
    
    def __init__(self,*args,**kwargs):
        
        super(MyConstantMean, self).__init__(*args,**kwargs)
        self.constant.requires_grad = False
    
    
    def set_train_data(self,local_mean,*args,**kwargs):
        
        self.constant.data = local_mean
        

    
    def append_train_data(self,new_mean,new_states,new_transitions):
        pass
        
class AdvantageMean(Mean):
    
    def __init__(self,agent):
        
        super(AdvantageMean, self).__init__()
        
        self.register_parameter(name="constant",parameter=torch.nn.Parameter(torch.zeros(1)))
        self.constant.requires_grad = False
        
        ## initialized later
        self.local_transitions = None
        self.local_states = None
        self.agent = agent

        #self.local_params = None (we directly use agent weights)
        
    
    def set_train_data(self,local_mean,local_states,local_transitions):
    
        self.constant.data = local_mean
        self.local_states = local_states
        self.local_transitions = local_transitions
        self.agent._replay_memory.add(local_transitions)
    
    def append_train_data(self,new_mean,new_states,new_transitions):
        
        self.agent._replay_memory.add(new_transitions)
    
    
    def fit_critic(self):
        
        ## add local transitions again just in case
        self.agent._replay_memory.add(self.local_transitions)
        self.agent.fit_critic()
        
    def __call__(self,params):



        
        
        with torch.no_grad():
        
            agent = self.agent
            local_params = agent._actor_approximator.model.network.default_weights.data
            local_actions = self.agent._actor_approximator(self.local_states,local_params).squeeze().T
            actions = self.agent._actor_approximator(self.local_states,params).squeeze().T
        
            local_q = agent._critic_approximator(self.local_states, local_actions, output_tensor=True, **agent._critic_predict_params)
            q = [agent._critic_approximator(self.local_states, a.T, output_tensor=True, **agent._critic_predict_params)
                
                for a in actions.T]
            
            q = torch.vstack(q)
                        
            return self.constant + torch.mean(q-local_q,axis=-1)
            
        
    
    def call2(self,theta_t):
     
        agent = self.agent
        actor = self.agent._actor_approximator.model.network
        
        """ Mushroom RL Call copies the parameters and breaks computation graph for grad"""
        #local_actions = self.agent._actor_approximator(self.local_states,theta_t).squeeze().T
        local_actions = actor(self.local_states,theta_t).squeeze().T
        local_q = agent._critic_approximator(self.local_states, local_actions,idx=0, output_tensor=True, **agent._critic_predict_params)
        #local_q = agent._target_critic_approximator(self.local_states, local_actions, output_tensor=True, **agent._critic_predict_params)
        #local_q = agent._target_critic_approximator(self.local_states, local_actions,idx=0,output_tensor=True, **agent._critic_predict_params)
        
        return torch.mean(local_q)
    
    