import gymnasium as gym
from environment import custom_gym

from tianshou.env import ShmemVectorEnv, VectorEnvNormObs


def make_custom_env(task, seed, training_num, test_num, obs_norm, render_mode=None):

    env = gym.make(task)

    train_envs = ShmemVectorEnv(
        [lambda: gym.make(task) for _ in range(training_num)]
    )
    test_envs = ShmemVectorEnv([lambda: gym.make(task) for _ in range(test_num)])

    train_envs.seed(seed)
    test_envs.seed(seed)


    if obs_norm:
        train_envs = VectorEnvNormObs(train_envs)
        test_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)
        test_envs.set_obs_rms(train_envs.get_obs_rms())


    return env, train_envs, test_envs


def make_mujoco_env(task, seed, training_num, test_num, obs_norm, render_mode=""):

    env = gym.make(task, render_mode=render_mode)
    train_envs = ShmemVectorEnv(
        [lambda: gym.make(task) for _ in range(training_num)]
    )
    if render_mode != "":
        test_envs = ShmemVectorEnv([lambda: gym.make(task, render_mode=render_mode) for _ in range(test_num)])
    else:
        test_envs = ShmemVectorEnv([lambda: gym.make(task) for _ in range(test_num)])

    train_envs.seed(seed)
    test_envs.seed(seed)
    if obs_norm:
        # obs norm wrapper
        train_envs = VectorEnvNormObs(train_envs)
        test_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)
        test_envs.set_obs_rms(train_envs.get_obs_rms())
    return env, train_envs, test_envs



