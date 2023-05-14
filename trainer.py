def train_func(env, q, episodes):
    print('----------train start----------')
    for episode in range(episodes):
        state, prob = env.reset()
        next_state = 0
        done = False
        step = 0
        success_num = 0
        total_reward = 0.
        while not done:
            # env.render()
            action = q.action(state)
            next_state, reward, done, info, _ = env.step(action)
            print(f'episode:{episode:<4d} step:{step:<4d} state:{state:<3d} action:{action} reward:{reward}')

            q.update(state=state, action=action, reward=reward, next_state=next_state)
            state = next_state
            total_reward += reward

            if reward > 0:
                success_num += 1
            step += 1

        print(f'episode:{episode:<4d} avg_reward:{total_reward / step}')
    print('q_table:\n', q.get_q_table())

    print('----------train end----------')


def eval_func(eval_env, q, eval_episodes):
    print('----------evaluate start----------')
    success_num = 0

    for episode in range(eval_episodes):
        state, prob = eval_env.reset()
        done = False
        while not done:
            action = q.get_max_action(state)
            next_state, reward, done, info, _ = eval_env.step(action)
            state = next_state

            if reward > 0:
                success_num += 1
    print(f'success rate={success_num / eval_episodes * 100}%')
    # print('q_table:\n', q.get_q_table())
    print('----------evaluate end----------')

    return success_num / eval_episodes
