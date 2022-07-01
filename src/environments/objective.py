from time import time
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import glfw
import gym
import numpy as np
import torch
from gym import wrappers


class EnvironmentObjective(object):
    """API for translating an OpenAI gym environment into a black-box objective
    function for which a parameterized mlp should be optimized.

    Attributes:
        env: OpenAI Gym environment.
        mlp: Parameterized mlp that should by optimized for env.
        manipulate_state: Function that manipluates the states of the
            environment.
        manipulate_reward: Function that manipluates the reward of the
            environment.
    """

   
    def __init__(
        self,
        env: gym.Env,
        mlp: Callable,
        reward_scale = 1.0,
        reward_shift = 0.0,
        *arg,**kwargs
    ):
        """Inits the translation environment to objective."""
        
        self.__dict__.update(locals())
        
        discrete = hasattr(env.info.action_space,"n")

        if discrete:
            self.mlp = self.discretize(mlp,num_actions=env.info.action_space.n)
        else:
            self.mlp = mlp
        
        
        self.horizon = env.info.horizon
        self.timesteps = 0
        self.timesteps_to_reward = {}
     
        
    def __call__(self, params,n_episodes=1) :
        return self.run_many(params,n_episodes)


    def run(
        self, params: torch.Tensor
    ) :
        """One rollout of an episodic environment with finite horizon.

        Evaluate value of current parameter constellation with sum of collected
        rewards over one rollout.

        Args:
            params: Current parameter constellation.
            render: If True render environment.
            test: If True renderer is not closed after one run.

        Returns:
            Cumulated reward.
        """
        states, actions, rewards,next_states,dones,lasts = [],[],[],[],[],[]
        r = 0
    
        next_state = self.env.reset()
        
        for t in range(self.horizon):  # rollout
            
            
            state = torch.Tensor(next_state)
            
            #### no need for grads here
            with torch.no_grad():
                
                if torch.is_tensor(params):
                    ### list of parameters for BO
                    action = self.mlp(params,state.unsqueeze(0)).squeeze()
                    #print("action shape",action.shape)
                
                else :
                    ### params is a torchregressor of mushroomrl
                    action = params(state.unsqueeze(0),output_tensor=True).squeeze()
                    #print("action shape",action.shape)
                    
            ###########################
            
            next_state, reward_tmp, done, _ = self.env.step(action.detach().numpy())
            
            last =  (t == (self.horizon-1)) or done

            
            states.append(torch.tensor(self.manipulate_state(state)))
            actions.append(action.detach())
            rewards.append(torch.tensor(self.manipulate_reward(reward_tmp)))
            next_states.append(torch.tensor(next_state))
            dones.append(torch.tensor(done))
            lasts.append(torch.tensor(last))
            
            
            if done:
                
                break
         
        
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)
        dones = torch.stack(dones)
        lasts = torch.stack(lasts)
        
        transitions = [
                        (s1.cpu().detach().numpy(),
                        a.cpu().detach().numpy(),
                        r.cpu().detach().numpy(),
                        s2.cpu().detach().numpy(),
                        d.cpu().detach().numpy(),
                        l.cpu().detach().numpy(),) 
                       for s1,a,r,s2,d,l in zip(states,actions,rewards,next_states,dones,lasts)
                       ] 
        
        return torch.sum(rewards),states,transitions
    
    def run_many(self, params,n_episodes):
       
        rewards = torch.tensor([0], dtype=torch.float32)
        all_states,all_transitions = [],[]
       
        for _ in range(n_episodes):
           
           reward,states,transitions = self.run(params)
           
           rewards += reward
           all_states.append(states)
           all_transitions +=(transitions)
        
        all_states = torch.cat(all_states)
        avg_reward = rewards/n_episodes
        
        return avg_reward,all_states,all_transitions
    
           
    def test_params(
        self,
        params: torch.Tensor,
        episodes: int,
        render: bool = True,
        path_to_video: Optional[str] = None,
        verbose: bool = True,
    ):
        """Test case for quantitative evaluation of parameter perfomance on
        environment.

        Args:
            params: Current parameter constellation.
            episodes: Number of episodes.
            render: If True render environment.
            path_to_video: Path to directory if a video wants to be saved.
            verbose: If True an output is logged.

        Returns:
            Cumulated reward.
        """
        if path_to_video is not None:
            self.env = wrappers.Monitor(self.env, path_to_video, force=True)
            import numpy as np
        num_digits = len(str(episodes))
        for episode in range(episodes):
            reward = self.run(params, render=render, test=True)
            #if verbose:
                #print(f"episode: {episode+1:{num_digits}}, reward: {reward}")
        if render:
            try:
                glfw.destroy_window(self.env.viewer.window)
                self.env.viewer = None
            except:
                self.env.close()
        return reward
    
  

    def manipulate_reward(self,reward):
        
            """Manipulate reward in every step with shift and scale.

            Args:
                shift: Reward shift.
                scale: Reward scale.

            Return:
                Manipulated reward.
            """
         
            return (reward - self.reward_shift) / self.reward_scale

    def manipulate_state(self,state):
        

        rslt = torch.tensor(state, dtype=torch.float32)
        
        return rslt



    def discretize(self,function: Callable, num_actions: int):
            """Discretize output/actions of MLP.
        For instance necessary for the CartPole environment.
        Args:
            function: Mapping states with parameters to actions, e.g. MLP.
            num_actions: Number of function outputs.
        Returns:
            Function with num_actions discrete outputs.
        """
        
            
            def discrete_policy_2(state, params):
                return (function(state, params) > 0.0) * 1

            def discrete_policy_n(state, params):
                return torch.argmax(function(state, params))

            if num_actions == 2:
                discrete_policy = discrete_policy_2
            elif num_actions > 2:
                discrete_policy = discrete_policy_n
            else:
                raise (f"Argument num_actions is {num_actions} but has to be greater than 1.")
            return discrete_policy


