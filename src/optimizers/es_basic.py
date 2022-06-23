
import numpy as np
import torch
import gym
from src.environment import EnvironmentObjective
from src.policy import MLPSequential
import pybullet_envs


class BasicNumGrad:
    def __init__(self, env, policy):
        self.env_obj = EnvironmentObjective(env, policy)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m = torch.zeros_like(policy.get_weights())
        self.v = torch.zeros_like(self.m)

    def adamize(self, grad, t):
        a = np.sqrt(1 - self.beta2 ** t) / (1 - self.beta1 ** t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad * grad)
        step = a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step

    def approx_grad(self, sigma=1e-1, roll_per_parms=10, param_per_step=10):
        mu = self.env_obj.mlp.get_weigths()
        all_e, all_f = [], []
        for k in range(param_per_step):
            all_e.append(torch.randn_like(mu) * sigma)
            self.env_obj.mlp.set_weights(mu + all_e[-1])
            all_f.append(self.env_obj.run_many(None, roll_per_parms)[0].item())

            # mirror perturbation
            all_e.append(-all_e[-1])
            self.env_obj.mlp.set_weights(mu + all_e[-1])
            all_f.append(self.env_obj.run_many(None, roll_per_parms)[0].item())

        nb_evals = len(all_f)

        # weights = np.asarray(all_f)
        # weights = (weights - weights.mean())  # also seems to do ok

        weights = np.argsort(all_f).argsort() / (nb_evals - 1) - .5

        return sum([w * e for w, e in zip(weights, all_e)]) / sigma / nb_evals, mu, all_f

    def optimize(self, step_size=1e-3, nb_steps=100, **approx_grad_kwargs):
        for k in range(nb_steps):
            grad, old_par, evals = self.approx_grad(**approx_grad_kwargs)
            print(f'iteration {k}, mean {sum(evals)/len(evals)} max {max(evals)}')
            adam_grad = self.adamize(grad, k + 1)
            self.env_obj.mlp.set_weights(old_par + step_size * adam_grad)


if __name__ == '__main__':
    torch.set_num_threads(1)
    # env = gym.make('Pendulum-v1')
    env = gym.make('Walker2DBulletEnv-v0')
    mlp = MLPSequential([env.observation_space.shape[0], 64, 64, env.action_space.shape[0]])

    # num grad
    optimizer = BasicNumGrad(env, mlp)
    optimizer.optimize(step_size=1e-3, nb_steps=500, sigma=2e-2, roll_per_parms=10, param_per_step=10)