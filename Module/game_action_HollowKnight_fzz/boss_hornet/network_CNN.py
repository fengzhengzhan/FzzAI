from dependencies import *

from argconfig import *

class DQN(nn.Module):
    def __init__(self, width, height, action):
        super(DQN, self).__init__()
        self.resize_dim = width * height
        self.resize_width = width
        self.resize_height = height
        self.mls = nn.MSELoss()


        # conv2d layer 图片卷积层
        self.conv = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=2),
            nn.BatchNorm2d(8),
            # nn.ReLU(inplace=True),
            nn.ReLU(),

            # nn.MaxPool2d(2, stride=1, ceil_mode=True),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
            # nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),

        )
        self.dropout = nn.Dropout

        # linear layer 线性层
        # 线性层的输入取决于conv2d的输出，计算输入图像大小
        self.linear_input_size = 14144

        self.linear = nn.Sequential(
            nn.Linear(self.linear_input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, action),
            nn.ReLU(),
        )

        # optimizer = optim.RMSprop(policy_net.parameters())
        self.opt = torch.optim.Adam(self.parameters(), lr=LEARN_RATE)


    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32).to(DEVICE).requires_grad_(True)
        x = self.conv(x)
        x = x.reshape(self.linear_input_size)
        x = self.linear(x)

        return x


class JUDGE():
    # 初始化class参数
    def __init__(self):
        self.replay_buffer = []
        self.batch_size = BATCH_SIZE
        self.action_size = ACTION_SIZE
        self.all_blood = 0
        self.all_power = 0
        self.choose_action_time = CHOOSE_ACTION_TIME  # 根据时间指导随机的操作选择引导

    # 经验回放
    def train_network(self, policy_net, target_net, num_step):
        # step 1: obtain random minibatch from replay memory!
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        # 从记忆库中采样BATCH_SIZE
        # print("[+] station_batch", station_batch, station_batch)
        # print("[+] action_batch", action_batch, action_batch)
        # print("[+] reward_batch", reward_batch, reward_batch)
        # print("[+] next_station_batch", next_station_batch, next_station_batch)
        q_batch = []
        tq_next_batch = []
        for i in range(0, self.batch_size):
            argm = minibatch[i][1].argmax()
            q = policy_net(minibatch[i][0]).detach()[argm]
            q_batch.append(q)
            q_next = target_net(minibatch[i][3]).detach().max()
            tq = minibatch[i][2] + GAMMA * q_next
            tq_next_batch.append(tq)
        q_batch = torch.as_tensor(q_batch, dtype=torch.float32)
        tq_next_batch = torch.as_tensor(tq_next_batch, dtype=torch.float32)

        loss = policy_net.mls(q_batch, tq_next_batch).requires_grad_(True)

        policy_net.opt.zero_grad()
        loss.backward()
        policy_net.opt.step()



    def choose_action(self, policy_net, station, choose_time):
        if random.random() >= (self.choose_action_time*5 - choose_time) / self.choose_action_time*6:
            return random.randint(0, self.action_size-1)
        else:
            state = policy_net(station).detach()
            return np.argmax(state)
    # 进行reward
    # 由于无法观测到boss血量，使用时间作为reward, reward为持续时间 tqt
    # 自身费血将会受到-reward
    #


    def action_judge(self, init_time, next_self_blood, next_self_power, action, stop, emergence_break):
        # 始终为0，stop=0 emergence_break=0为预留参数，借鉴B站DQN训练只狼的代码中遗留部分，未删减
        return_reward = 0
        stop = 0
        if next_self_blood < 1:  # 自己死亡
            if emergence_break < 2 and self.all_blood >= 3:
                reward = 1
                done = 1
                emergence_break = 0
                return init_time, reward, done, stop, emergence_break
            elif emergence_break < 2:
                reward = -0.8
                done = 1
                emergence_break = 0
                return init_time, reward, done, stop, emergence_break
            else:
                reward = -0.8
                done = 1
                emergence_break = 100
                return init_time, reward, done, stop, emergence_break

        done = 0
        emergence_break = 0


        if self.all_blood - next_self_blood >= 1:  # 费血
            if emergence_break < 2:
                return_reward += -0.5 * (self.all_blood - next_self_blood)
                self.all_blood = next_self_blood
            else:
                return_reward += -0.5 * (self.all_blood - next_self_blood)
                self.all_blood = next_self_blood
                emergence_break = 100
        elif next_self_blood - self.all_blood >= 1:  # 回血
            if emergence_break < 2:
                return_reward += 0.6 * (next_self_blood - self.all_blood)
                self.all_blood = next_self_blood
            else:
                return_reward += 0.6 * (next_self_blood - self.all_blood)
                self.all_blood = next_self_blood
                emergence_break = 100

        # 早期进行攻击也奖励，已取消
        # if action == I_ATTACK or action == J_LEFT or action == K_ATTACK or action == L_RIGHT or action == ATTACK_NUM:  # 攻击给予正反馈
        #     if emergence_break < 2:
        #         return_reward += 8
        #     else:
        #         return_reward += 8
        #         emergence_break = 100

        if next_self_power - self.all_power >= 5:  # 增加能量
            if emergence_break < 2:
                return_reward += 0.4
                self.all_power = next_self_power
            else:
                return_reward += 0.4
                self.all_power = next_self_power
                emergence_break = 100

        if self.all_power - next_self_power >= 4:
            self.all_power = next_self_power

        # 早期增加时间也给予奖励，已经取消
        # if emergence_break < 2:
        #     return_reward += time.time() - init_time
        #     init_time = time.time()
        # else:
        #     return_reward += time.time() - init_time
        #     init_time = time.time()
        #     emergence_break = 100

        return init_time, return_reward, done, stop, emergence_break


    #列表存储
    def store_data(self, station, action, reward, next_station):
        one_hot_action = np.zeros(self.action_size)
        one_hot_action[action] = 1
        self.replay_buffer.append((station, one_hot_action, reward, next_station))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.pop(0)

if __name__ == '__main__':
    pass

import time
import random
from argconfig import *
import pickle
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T

# 将REPLAY_BUFFER经验回放的存储库弹出一部分
REPLAY_BUFFER = []
if os.path.exists(DQN_STORE_PATH):
    REPLAY_BUFFER = pickle.load(open(DQN_STORE_PATH, 'rb'))
for i in range(600):
    REPLAY_BUFFER.pop(len(REPLAY_BUFFER)-1)
pickle.dump(REPLAY_BUFFER, open(DQN_STORE_PATH, 'wb'))
print(REPLAY_BUFFER, type(REPLAY_BUFFER), len(REPLAY_BUFFER))

