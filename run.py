import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import config
from algo.q_learning import QLearing
from algo.sarsa import Sarsa
from algo.demo_policy import DemoPolicy
from trainer import train_func, eval_func
import numpy as np

np.set_printoptions(suppress=True)


def init_env():
    is_render = config.env_config['render']
    if config.env_config['random_map']:
        _env = gym.make(config.env_config['name'], desc=generate_random_map(size=config.env_config['map_width']),
                        is_slippery=config.env_config['is_slippery'],
                        render_mode=config.env_config['render_mode'] if is_render else None)
    else:
        _env = gym.make(config.env_config['name'], desc=None, map_name=config.env_config['map_name'],
                        is_slippery=config.env_config['is_slippery'],
                        render_mode=config.env_config['render_mode'] if is_render else None)
    _env.reset()
    return _env


def run_q_learning():
    print(config.env_config)
    print(config.train_config)
    _q_learning = QLearing(config.env_config, config.train_config)

    env = init_env()
    eval_env = init_env()

    episodes = config.train_config['episodes']
    eval_episodes = config.train_config['eval_episodes']

    # for _ in range(episodes // 500):
    train_func(env, _q_learning, episodes)
    success_rate = eval_func(eval_env, _q_learning, eval_episodes)

    env.close()
    eval_env.close()

    return success_rate


def run_sarsa():
    print(config.env_config)
    print(config.train_config)
    _sarsa = Sarsa(config.env_config, config.train_config)

    env = init_env()
    eval_env = init_env()

    episodes = config.train_config['episodes']
    eval_episodes = config.train_config['eval_episodes']

    train_func(env, _sarsa, episodes)
    success_rate = eval_func(eval_env, _sarsa, eval_episodes)

    env.close()
    eval_env.close()

    return success_rate

# run_q_learning()
# run_sarsa()
