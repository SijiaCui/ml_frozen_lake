env_config = {
    'random_map': False,
    'map_name': "4x4",
    'map_width': 4,

    'is_slippery': True,
    'render': False,

    # fixed params
    'name': 'FrozenLake-v1',
    'render_mode': 'human',
    'action_space': 4,
}

train_config = {
    'episodes': 1000,
    'eval_episodes': 1000,  # fixed
    'learning_rate': 0.5,
    'gamma': 0.99,

    'use_epsilon': True,
    'epsilon': 1.0,
    # 1 is linear decay, 2 is exp decay
    'epsilon_decay_mode': 1,
    'linear_epsilon_decay': 1 / 5000,
    'exp_epsilon_decay': 0.998,

}

# print(env_config, train_config)
