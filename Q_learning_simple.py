import numpy as np
import pandas as pd
import time

np.random.seed(2)   # 固定随机

N_STATES = 6    # 总状态个数
ACTIONS = ['left', 'right']
EPSILON = 0.9     # 贪心率
ALPHA = 0.1     # 学习率
GAMMA = 0.9
MAX_EPISODES = 13  # 最大场数
FRESH_TIME = 0.31  # 每个动作执行时间


# Q table
def build_q_table(nstates, actions):
    table = pd.DataFrame(
        np.zeros((nstates, len(actions))),
        columns=actions
    )
    print('q table\n', table)
    return table


# 动作选择
def choose_action(state, q_table):
    state_action = q_table.iloc[state, :]   # 表内数值是各个动作的q值
    # 随机选择,概率0-1 或者在全0初始化的时候
    if (np.random.uniform() > EPSILON) or (state_action.all() == 0):
        action_name = np.random.choice(ACTIONS)    # 随机选一个动作
    else:
        action_name = state_action.idxmax()   # 选择q值最大的那个列的label
    return action_name


# 环境反馈
def get_env_feedback(S, A):
    """
    这里没有设置马尔可夫状态转移概率
    :param S: 当前状态
    :param A: 打算做的动作
    :return: 下个状态，得到的奖励
    """
    if A == 'right':
        if S == N_STATES - 2:  # 到达终点
            S_ = 'terminal'
            R = 1   # reward
        else:
            S_ = S + 1   # 向右移动
            R = 0
    else:
        # 向左移动
        R = 0
        if S == 0:
            S_ = S   # 到达边界原地不动
        else:
            S_ = S -1
    return S_, R


# 创建环境，只负责显示，和算法本身无关
def update_env(S, episode, step_counter):
    env = ['-']*(N_STATES-1)+['T']  # 搭建环境
    if S == 'terminal':
        interaction = 'Episode %s: total_steps= %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r         ', end='')    # 覆盖上一行的显示
    else:
        env[S] = '0'
        interaction = ''.join(env)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    """
        和环境交互，得到新的q表
        Q(新) <- Q(旧)+alpha*
    :return:
    """
    table = build_q_table(N_STATES, ACTIONS)   # 创建表
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0     # 初始状态
        is_terminated = False
        update_env(S, episode, step_counter)  # 更新环境
        while not is_terminated:
            A = choose_action(S, table)  # 选q值大的行为
            S_, R = get_env_feedback(S, A)  # 实施行为并得到环境的反馈

            # 得到采取A后的预测q值
            q_predict = table.loc[S, A]  # 估算的(状态-行为)值
            if S_ == 'terminal':
                q_target = R  # 实际的(状态-行为)值 (回合结束)
                is_terminated = True  # 本场游戏终止
            else:
                # A动作后的真实Q值
                q_target = R + GAMMA * table.iloc[S_, :].max()  # 实际的(状态-行为)值 (回合没结束)

            table.loc[S, A] += ALPHA * (q_target - q_predict)  # q_table 更新
            S = S_  # 探索者移动到下一个 state

            update_env(S, episode, step_counter + 1)  # 环境更新

            step_counter += 1    # 花费步数
    return table


if __name__ == "__main__":
    q_table = rl()
    print('Q-table')
    print(q_table)
