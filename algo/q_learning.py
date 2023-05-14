import random
import numpy as np


class QLearing:
    def __init__(self, env_params, q_learning_params, init_q_table=None):
        self.step = 0
        self.state_space = env_params['map_width'] ** 2
        self.action_space = env_params['action_space']
        self.q_table = np.zeros((self.state_space, self.action_space))
        self.lr = q_learning_params['learning_rate']
        self.gamma = q_learning_params['gamma']

        self.use_epsilon = q_learning_params['use_epsilon']
        if self.use_epsilon:
            self.epsilon = q_learning_params['epsilon']
            self.epsilon_decay_mode = q_learning_params['epsilon_decay_mode']
            self.linear_epsilon_decay = q_learning_params['linear_epsilon_decay']
            self.exp_epsilon_decay = q_learning_params['exp_epsilon_decay']

    def get_q_table(self):
        return self.q_table

    def update(self, state, action, reward, next_state):
        self.q_table[state, action] = self.q_table[state, action] + self.lr * (
                reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state, action])

    def action(self, s):
        if self.use_epsilon:
            rnd = np.random.random()
            if rnd < self.epsilon:
                action = random.randint(0, self.action_space - 1)
            else:
                action = self.get_max_action(s)

            # epsilon decay
            if self.epsilon_decay_mode == 1:
                self.epsilon = max(self.epsilon - self.linear_epsilon_decay, 0)
            elif self.epsilon_decay_mode == 2:
                self.epsilon *= self.epsilon_decay_mode
            else:
                print('WARNING: use epsilon, but dont use any decay')
                assert False
        else:
            action = self.get_max_action(s)
        return action

    def get_max_action(self, s):
        if np.max(self.q_table[s]) > 0:
            return np.argmax(self.q_table[s])
        else:
            return random.randint(0, self.action_space - 1)
