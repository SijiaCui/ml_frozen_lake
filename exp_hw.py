import config
from run import run_q_learning, run_sarsa
import sys
import os


def q_learning_lr_experiment(lrs=None):
    lrs = [0.1, 0.3, 0.5, 0.7, 0.9] if not lrs else lrs
    srs = []
    sys.stdout = open(os.devnull, 'w')
    for _lr in lrs:
        config.train_config['learning_rate'] = _lr
        success_rate = 0
        for i in range(10):
            success_rate += run_q_learning()
        srs.append(success_rate / 10)
    sys.stdout = sys.__stdout__

    print('q learning with different lrs: ', lrs)
    print('average success rate: ', srs)


def q_learning_gamma_experiment(gammas=None):
    gammas = [0., 0.5, 0.9, 0.99, 0.999] if not gammas else gammas
    srs = []
    sys.stdout = open(os.devnull, 'w')
    for _gamma in gammas:
        config.train_config['gamma'] = _gamma
        success_rate = 0
        for i in range(10):
            success_rate += run_q_learning()
        srs.append(success_rate / 10)
    sys.stdout = sys.__stdout__

    print('q learning with different gammas: ', gammas)
    print('average success rate: ', srs)


def sarsa_lr_experiment(lrs=None):
    lrs = [0.1, 0.3, 0.5, 0.7, 0.9] if not lrs else lrs
    srs = []
    sys.stdout = open(os.devnull, 'w')
    for _lr in lrs:
        config.train_config['learning_rate'] = _lr
        success_rate = 0
        for i in range(10):
            success_rate += run_sarsa()
        srs.append(success_rate / 10)
    sys.stdout = sys.__stdout__

    print('sarsa with different lrs: ', lrs)
    print('average success rate: ', srs)


def sarsa_gamma_experiment(gammas=None):
    gammas = [0., 0.5, 0.9, 0.99, 0.999] if not gammas else gammas
    srs = []
    sys.stdout = open(os.devnull, 'w')
    for _gamma in gammas:
        config.train_config['gamma'] = _gamma
        success_rate = 0
        for i in range(10):
            success_rate += run_sarsa()
        srs.append(success_rate / 10)
    sys.stdout = sys.__stdout__

    print('sarsa with different gammas: ', gammas)
    print('average success rate: ', srs)


q_learning_lr_experiment()
q_learning_gamma_experiment()
sarsa_lr_experiment()
sarsa_gamma_experiment()
