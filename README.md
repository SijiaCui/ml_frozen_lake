## 项目结构
![](.\project.png)

## 主要工作
实现Frozen Lake环境下的Q-Learning和Sarsa算法。

在is_slippery=false的情况下，达到100%的成功率。详见`run_qlearning.log`和`run_sarsa.log`文件。

在is_slippery=true的情况下，达到80+%成功率(4x4地图)。详见`run_qlearning_slippery.log`和`run_sarsa_slippery.log`文件。

对于不同环境，不同超参数的训练，只需要修改config.py文件，运行`python run.py > XXX.log`

## Config说明
- `random_map`如果为`True`，则会调用`generate_random_map`随机生成`map_width`尺寸的图。
如果为`False`，则按照`map_name`获得图。

- `is_slippery`用来设置移动是否打滑，详见[官网的解释](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/)。

- `render`表示是否选择可视化。

```
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
```

- `episodes`和`eval_episodes`分别表示训练数和评估时的测试数。

- `use_epsilon`表示选择动作时，是否使用\epsilon-greedy方法。 Sarsa需设置为`True`。
`epsilon_decay_mode`表示如果使用\epsilon-greedy方法，采用decay的模式，0表示不使用任何decay方法，1表示使用线性decay，2表示使用指数decay。
相应的参数为`linear_epsilon_decay`和`exp_epsilon_decay`。


```
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
```
