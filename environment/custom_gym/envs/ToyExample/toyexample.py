import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from scipy.stats import norm

INTERMEDIATE_REWARD = 0
FINAL_POSITIVE_REWARD = 1
FINAL_NEGATIVE_REWARD_G = -20
FINAL_NEGATIVE_REWARD_B = -1

ACTION_BOUND_LOW = -0.1
ACTION_BOUND_HIGH = 0.1

class ToyEnv(gym.Env):
    def __init__(self):
        super(ToyEnv, self).__init__()
        self.num_labels = 6
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_labels,), dtype=np.float32)

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            dtype=np.float32
        )
        self.state = 0
        self.action_bound_low = ACTION_BOUND_LOW
        self.action_bound_high = ACTION_BOUND_HIGH
        self.seed()

    def get_observation(self):
        obs = np.zeros(6, dtype=np.float32)
        obs[self.state] = 1.
        return obs


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, **kwargs):
        self.state = 0
        obs = self.get_observation()

        return obs, {}

    def step(self, action):
        reward = INTERMEDIATE_REWARD
        done = False

        if self.state == 0:
            if action > 0:
                self.state = 2
            else:
                self.state = 1

        elif self.state == 1:
            if self.action_bound_low < action < self.action_bound_high:
                self.state = 3
            else:
                self.state = 4

        elif self.state == 2:
            self.state = 5

        if self.state == 3:
            done = True
            reward = FINAL_POSITIVE_REWARD

        elif self.state == 4:
            done = True
            reward = FINAL_NEGATIVE_REWARD_G

        elif self.state == 5:
            done = True
            reward = FINAL_NEGATIVE_REWARD_B

        obs = self.get_observation()

        return obs, reward, done, False, {}

    def reset_by_idx(self, x):
        self.state = x
        return self.get_observation()

    def get_obs_by_idx(self, x):
        obs = np.zeros(6, dtype=np.float32)
        obs[x] = 1.
        return obs


if __name__ == "__main__":
    env = ToyEnv()
    obs = env.reset()
    print(obs)
