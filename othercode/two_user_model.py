#数学基本库，引入并重命名为np
import numpy as np
#数据分析库
import pandas as pd
import math

P = 100   #总功率
yinzi = 0    #功率分配因子
N_STATES = 11   # 状态个数
ACTIONS = ['+0.1', '-0.1']     # 探索者的可用动作，在一维世界只有左右
EPSILON = 0.9   # 贪婪度，即探索者有90%的情况会按照Q表的最优值选择行为，10%的时间会随机选择行为
ALPHA = 0.1     # 学习率，用来决定误差有多少需要被学习的，ALPHA是一个小于1的数
GAMMA = 0.9    # 奖励递减值，表示对未来reward的衰减值
MAX_EPISODES = 30   # 最大回合数
epoch = 100 #迭代轮数
h1 = 1
h2 = 2
N0 = 100

#DataFrame是Python中Pandas库中的一种数据结构，它类似excel，是一种二维表。
#建立q表
def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table为一个6x2的表格，并初始化值都为0
        columns=actions,    # actions's name
    )
    # print(table)    # show table
    '''
	    -0.1  +0.1
	0   0.0    0.0
	1   0.0    0.0
	2   0.0    0.0
	3   0.0    0.0
	4   0.0    0.0
	5   0.0    0.0
	'''
    return table


def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]   # iloc函数提取行数据（对数据进行位置索引，从而在数据表中提取出相应的数据
    #np.random.uniform() 随机分布0-1
    if((np.random.uniform() < EPSILON) or ((state_actions == 0).all())):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()
    return action_name


def get_env_feedback(S, A):
    # 探索者与环境互动，由当前的状态与动作获取奖励及探索者的下一状态
    global yinzi
    if A == '+0.1':
        yinzi = yinzi + 0.1
        if yinzi > 1:
            yinzi -= 0.1
            S_ = S
        else:
            S_ = S + 1
        p1 = P * yinzi
        sinr1 = p1 * h1 ** 2 / ((P-p1) * h1 ** 2) + N0
        R1 = 1/2 * math.log2(1+sinr1)
        p2 = P - p1
        sinr2 = p2 * h2 ** 2 / N0
        R2 = 1/2 * math.log2(1 + sinr2)
    else:   # 探索者当前状态下向左移动
        yinzi = yinzi - 0.1
        if yinzi < 0:
            yinzi += 0.1
            S_ = S
        else:
            if S == 0:
                S_ = 0
            else:
                S_ = S - 1

            # if S == 0:
            #     S_ = S - 1
            # else:
            #     S_ = S - 1
        p1 = P * yinzi
        sinr1 = p1 * h1 ** 2 / ((P - p1) * h1 ** 2) + N0
        R1 = (1 / 2) * math.log2(1 + sinr1)
        p2 = P - p1
        sinr2 = p2 * h2 ** 2 / N0
        R2 = (1 / 2) * math.log2(1 + sinr2)
    R = R1 + R2
    print(yinzi)
    return S_, R

def rl():
    # q-learning的主要部分
    q_table = build_q_table(N_STATES, ACTIONS)  # 建立一个q表，且初始值都为0
    # print(q_table)

    for episode in range(MAX_EPISODES):  # 训练 MAX_EPISODES 个回合
        S = 0  # 初始化探索者的状态（位置）
        for d in range(epoch):  # 探索者进行探索
            A = choose_action(S, q_table)  # 选择动作
            S_, R = get_env_feedback(S, A)  # 根据当前的状态及动作，获取下一状态和奖励
            # print(S_)
            q_predict = q_table.loc[S, A]
            # try:
            #     q_predict = q_table.loc[S, A]   # 根据当前的状态及动作，取出当前q表中对应位置的值，即Q估计
            # except Exception as e:
            #     print("error:", e, "\n", "value:", S, A)
            #     continue

            q_target = R + GAMMA * q_table.iloc[S_, :].max()   # 计算下一状态不是 terminal 时的 Q现实
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # 对q表进行更新
            S = S_  # move to next state

    print('\r\nQ-table:\n')
    print(q_table)

if __name__ == '__main__':
    rl()

    #KeyError: -1