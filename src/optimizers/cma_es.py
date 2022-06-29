import cma
import torch
import gym
from src.environment import EnvironmentObjective
from src.approximators.actor import MLPSequential
import pybullet_envs


class CMARL:
    def __init__(self, env, policy, sigma, roll_per_par):
        self.env_obj = EnvironmentObjective(env, policy)
        self.roll_per_par = roll_per_par
        self.optimizer = cma.CMAEvolutionStrategy(policy.get_weights().detach().numpy(), sigma0=sigma,
                                                  inopts={'CMA_diagonal': True})

    def objective_fn(self, theta):
        self.env_obj.mlp.set_weights(torch.tensor(theta, dtype=torch.float))
        return -self.env_obj.run_many(None, self.roll_per_par)[0].item()

    def optimize(self):
        self.optimizer.optimize(self.objective_fn, maxfun=100000)


if __name__ == '__main__':
    torch.set_num_threads(1)
    # env = gym.make('Pendulum-v1')
    env = gym.make('Walker2DBulletEnv-v0')
    mlp = MLPSequential([env.observation_space.shape[0], 64, 64, env.action_space.shape[0]])

    # num grad
    optimizer = CMARL(env, mlp, sigma=.1, roll_per_par=10)
    optimizer.optimize()
