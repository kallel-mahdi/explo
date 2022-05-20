from typing import Tuple, Dict, Callable, Iterator, Union, Optional, List

#from abc import ABC, abstractmethod

from time import time
import glfw
import torch
import gym
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

        # if manipulate_reward is None:
        #     manipulate_reward = lambda reward, action, state, done: reward
        # self.manipulate_reward = manipulate_reward
        
        self.manipulate_reward = manipulate_reward(reward_shift,reward_scale)

        self._manipulate_state = manipulate_state
        if manipulate_state is None:
            manipulate_state = lambda state: state

        self.manipulate_state = lambda state: manipulate_state(
            torch.tensor(state, dtype=dtype_states)
        )

    def __call__(self, params: torch.Tensor) :
        return self.run(params)

    def _unpack_episode(self):
        
        """Helper function for get_last_episode.

        Get states, actions and rewards of last episode.

        Returns:
            Tuple of states, actions and rewards.
        """
        states = self._last_episode["states"]
        actions = self._last_episode["actions"]
        rewards = self._last_episode["rewards"]
        return states, actions, rewards
    

    def get_last_episode(self):
        """Return states, actions and rewards of last episode.
        Implemented for the implementation of mlp gradient methods.

        Returns:
            Dictionary of states, actions and rewards.
        """
        
        states, actions, rewards = self._unpack_episode()
        return {
            "states": states[: self._last_episode_length + 1].clone(),
            "actions": actions[: self._last_episode_length].clone(),
            "rewards": rewards[: self._last_episode_length].clone(),
        }

    def run(
        self, params: torch.Tensor, render: bool = False, test: bool = False
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
        states, actions, rewards = self._unpack_episode()
        r = 0
        states[0] = self.manipulate_state(self.env.reset())
        
        for t in range(self.max_steps):  # rollout
            
            #### no need for grads here
            with torch.no_grad():
                actions[t] = self.mlp(params,states[t].unsqueeze(0)).squeeze()
                #actions[t] = self.mlp(params,states[t])
            ###########################
            
            state, rewards[t], done, _ = self.env.step(actions[t].detach().numpy())
            states[t + 1] = self.manipulate_state(state)
            r += self.manipulate_reward(
                rewards[t], actions[t], states[t + 1], done
            )  # Define as stochastic gradient ascent.
            if render:
                self.env.render()
            if done:
                break
        if not test:
            self.timesteps += t
            self.timesteps_to_reward.update({self.timesteps: rewards[:t].sum()})
        self._last_episode_length = t
        if render and not test:
            self.env.close()
            
        return torch.tensor([r], dtype=torch.float32),states
    
   

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
        num_digits = len(str(episodes))
        for episode in range(episodes):
            reward = self.run(params, render=render, test=True)
            if verbose:
                print(f"episode: {episode+1:{num_digits}}, reward: {reward}")
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
    return lambda reward, action, state, done: (reward - shift) / scale