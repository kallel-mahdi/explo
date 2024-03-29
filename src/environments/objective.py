from time import time
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union
from mushroom_rl.utils.spaces import Discrete 

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
        manipulate_state,
        reward_scale = 1.0,
        reward_shift = 0.0,
        discount_factor = 0.99,
        *arg,**kwargs
    ):
        """Inits the translation environment to objective."""
        
        self.__dict__.update(locals())
        
        discrete = isinstance(env.info.action_space,Discrete)

        if discrete:
            print("discritizing action space")
            self.mlp = self.discretize(mlp,num_actions=env.info.action_space.n)
        else:
            self.mlp = mlp
        
        self.horizon = env.info.horizon
        self.gamma = torch.tensor(discount_factor)
        self.timesteps_to_reward = {}
        self.timesteps = 0
        
        
        if manipulate_state == False:
            self.manipulate_state = lambda state: state

        elif manipulate_state == True:
            
            print(f'Using state normalization')
            
            self.state_normalizer = StateNormalizer()
            self.manipulate_state = lambda state: self.state_normalizer(state)
     
        
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
        states, actions, rewards,discounts,next_states,dones,lasts = [],[],[],[],[],[],[]
        r = 0
    
        next_state = self.env.reset()
        disc = torch.tensor(1.0)
        
        for t in range(self.horizon):  # rollout
            
            
            state = torch.Tensor(self.manipulate_state(next_state))
            
            #### no need for grads here
            with torch.no_grad():
                
                if torch.is_tensor(params):
                    ### list of parameters for BO
                    action = self.mlp(state.unsqueeze(0),params).squeeze()
                
                else :
                    ### params is a torchregressor of mushroomrl
                    action = params(state.unsqueeze(0),output_tensor=True).squeeze()
                    
                    
                    
            ###########################
            
            next_state, reward_tmp, done, _ = self.env.step(action.detach().numpy())
            
            last =  (t == (self.horizon-1)) or done

            
            states.append(state)
            actions.append(action.detach())
            rewards.append(torch.tensor(self.manipulate_reward(reward_tmp)))
            discounts.append(disc)
            next_states.append(torch.tensor(self.manipulate_state(next_state))) ## next_state is not normalized !!
            dones.append(torch.tensor(done))
            lasts.append(torch.tensor(last))
            
            # update discount factor
            disc *= self.gamma            
            
            
            if done:
                
                break
         
        
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        discounts = torch.stack(discounts)
        next_states = torch.stack(next_states)
        dones = torch.stack(dones)
        lasts = torch.stack(lasts)
        
        transitions = [
                        (s1.cpu().detach().numpy(),
                        a.cpu().detach().numpy(),
                        r.cpu().detach().numpy(),
                        disc.cpu().detach().numpy(),
                        s2.cpu().detach().numpy(),
                        d.cpu().detach().numpy(),
                        l.cpu().detach().numpy(),) 
                       for s1,a,r,disc,s2,d,l in zip(states,actions,rewards,discounts,next_states,dones,lasts)
                       ] 
        
        return torch.sum(rewards),torch.sum(discounts*rewards),states,transitions
    
    def run_many(self, params,n_episodes):
       
        rewards = torch.tensor([0], dtype=torch.float32)
        all_states,all_transitions,all_rewards,all_disc_rewards = [],[],[],[]
        
       
        for _ in range(n_episodes):
           
           reward,disc_reward,states,transitions = self.run(params)
           
           all_states.append(states)
           all_transitions +=(transitions)
           all_rewards.append(reward)
           all_disc_rewards.append(disc_reward)
        
        all_states = torch.cat(all_states)
        avg_reward = torch.mean(torch.stack(all_rewards))
        avg_disc_reward = torch.mean(torch.stack(all_disc_rewards))
        var_reward = torch.tensor(all_rewards).var()
        
        return avg_reward.reshape(1).float(),avg_disc_reward.reshape(1).float(),all_states,all_transitions,var_reward
    
           
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
        

        return state



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



class StateNormalizer:
    """Class for state normalization.

    Implementation of Welfords online algorithm. For further information see
        thesis appendix A.3.

    Attributes:
        eps: Small value to prevent division by zero
        normalize_params: Normalization function for policy parameters.
        unnormalize_params: Unnormalization function for policy parameters.
    """

    def __init__(
        self, eps: float = 1e-8, normalize_params=None, unnormalize_params=None
    ):
        # Init super.
        self.eps = eps
        self.steps = 0
        self._mean_of_states = 0.0
        self._sum_of_squared_errors = 0.0
        self.mean = 0.0
        self.std = 1.0
       
    def _welford_update(self, state: torch.Tensor):
        """Helper function for manipulate.

        Internally trackes mean and std according to the seen states.

        Args:
            state: New state.
        """
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        self.steps += 1
        delta = state - self._mean_of_states
        self._mean_of_states += delta / self.steps
        self._sum_of_squared_errors += delta * (state - self._mean_of_states)
    
 
    def manipulate(self, state: torch.Tensor) -> torch.Tensor:
        
        """Actually manipulate a state with the tracked mean and standard
        deviation.

        Args:
            state: State to normalize.

        Returns:
            Normalized state.
        """
        self._welford_update(state)
        normalized_state = (state - self.mean) / self.std
        return normalized_state
    
    def __call__(self, state):
        return self.manipulate(state)

