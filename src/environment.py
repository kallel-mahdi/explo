from time import time
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import glfw
import gym
import numpy as np
import torch
from gym import wrappers


class EnvironmentObjective:
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
        manipulate_state: Optional[Callable] = None,
        #manipulate_reward: Optional[Callable] = None,
        *arg,**kwargs
    ):
        """Inits the translation environment to objective."""
        self.env = env
        
        discrete = hasattr(env.action_space,"n")

        if discrete:
            self.mlp = discretize(mlp,num_actions=env.action_space.n)
        else:
            self.mlp = mlp
        
        
        self.max_steps = env._max_episode_steps
        self.timesteps = 0
        self.timesteps_to_reward = {}
        shape_states = env.observation_space.shape
        dtype_states = torch.float32

        shape_actions = env.action_space.shape
        dtype_actions = torch.tensor(env.action_space.sample()).dtype

        self._last_episode_length = 0
        self._last_episode = {
            "states": torch.empty(
                (self.max_steps + 1,) + shape_states, dtype=dtype_states
            ),
            "actions": torch.empty(
                (self.max_steps,) + shape_actions, dtype=dtype_actions
            ),
            "rewards": torch.empty(self.max_steps, dtype=torch.float32),
        }

        
        self.manipulate_reward = manipulate_reward(reward_shift,reward_scale)

        self._manipulate_state = manipulate_state
        if manipulate_state is None:
            manipulate_state = lambda state: state

        self.manipulate_state = lambda state: manipulate_state(
            torch.tensor(state, dtype=dtype_states)
        )

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
        states, actions, rewards = [],[],[]
        r = 0
        states.append(self.manipulate_state(self.env.reset()))
        
        for t in range(self.max_steps):  # rollout
            
            #### no need for grads here
            with torch.no_grad():
                
                # action = self.mlp(params,states[t].unsqueeze(0)).squeeze()                
                if params is not None:
                    action = self.mlp(params,states[t].unsqueeze(0)).squeeze(0)
                else:
                    action = self.mlp(states[t].unsqueeze(0)).squeeze(0)
                
            ###########################
            
            state, reward_tmp, done, _ = self.env.step(action.detach().numpy())
            
            rewards.append(self.manipulate_reward(reward_tmp))
            states.append(self.manipulate_state(state))
            actions.append(action)
            
            
            if done:
                
                break
         
         
        rewards = torch.tensor(rewards)
        actions = torch.stack(actions)
        states = torch.stack(states)


        #print(f'one episode : actions {actions.shape} / states {states.shape}')
        
        return torch.sum(rewards),states
    
    def run_many(self, params,n_episodes):
       
        rewards = torch.tensor([0], dtype=torch.float32)
        all_states = []
       
        for _ in range(n_episodes):
           
           reward,states = self.run(params)
           
           rewards += reward
           all_states.append(states)
           
        all_states = torch.cat(all_states)     
        avg_reward = rewards/n_episodes
        
        #print(f' avg_reward{avg_reward}, all_states{all_states.shape}')
        
        return avg_reward,all_states
    
           
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



def discretize(function: Callable, num_actions: int):
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

def manipulate_reward(shift: Union[int, float], scale: Union[int, float]):
    """Manipulate reward in every step with shift and scale.

    Args:
        shift: Reward shift.
        scale: Reward scale.

    Return:
        Manipulated reward.
    """
    if shift is None:
        shift = 0
    if scale is None:
        scale = 1
    return lambda reward: (reward - shift) / scale
