import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import os
import random
import numpy as np

from argconfig import *


# class Bottleneck(nn.Module):
#     def __init__(self, in_channels, filters, stride=1, is_downsample=False):
#         super(Bottleneck, self).__init__()
#         filter1, filter2, filter3 = filters
#         self.conv1 = nn.Conv2d(in_channels, filter1, kernel_size=1, stride=stride, bias=False)
#         self.bn1 = nn.BatchNorm2d(filter1)
#         self.conv2 = nn.Conv2d(filter1, filter2, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(filter2)
#         self.conv3 = nn.Conv2d(filter2, filter3, kernel_size=1, stride=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(filter3)
#         self.relu = nn.ReLU(inplace=True)
#         self.is_downsample = is_downsample
#         self.parameters()
#         if is_downsample:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_channels, filter3, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(filter3),
#             )
#
#     def forward(self, x):
#         x_shortcut = x
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#
#         x = self.conv3(x)
#         x = self.bn3(x)
#
#         if self.is_downsample:
#             x_shortcut = self.downsample(x_shortcut)
#
#         x = x + x_shortcut
#         x = self.relu(x)
#         return x


class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
        )

        self.shortcut = nn.Sequential()
        if ch_in != ch_out or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = self.block(x)
        out = self.shortcut(x) + out
        out = F.relu(out)

        return out


class DQN(nn.Module):
    def __init__(self, move_action, attack_action):
        super(DQN, self).__init__()
        # self.resize_dim = width * height
        # self.resize_width = width
        # self.resize_height = height
        self.mls = nn.MSELoss()
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        self.blk1 = ResBlk(8, 12, stride=2)
        self.blk2 = ResBlk(12, 16, stride=2)
        self.blk3 = ResBlk(16, 24, stride=2)
        self.blk4 = ResBlk(24, 32, stride=2)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # linear layer 线性层
        # 线性层的输入取决于conv2d的输出，计算输入图像大小
        self.linear_input_size = 1920

        self.move = nn.Sequential(
            nn.Linear(self.linear_input_size, 256),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Dropout(p=0.8),
            nn.Linear(128, move_action),
        )

        self.attack = nn.Sequential(
            nn.Linear(self.linear_input_size, 256),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Dropout(p=0.8),
            nn.Linear(128, attack_action),
        )

        self.opt = torch.optim.Adam(self.parameters(), lr=LEARN_RATE)



    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32).to(DEVICE)

        # x = torch.as_tensor(x, dtype=torch.float32).to(DEVICE)
        # x = F.max_pool2d(x, (2, 2))
        # nn.MaxPool2d(2, stride=2, ceil_mode=True),
        # print(x.shape)
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        # x = self.conv(x)

        # x = self.avgpool(x)
        x = torch.flatten(x, 1)

        move = self.move(x)
        attack = self.attack(x)

        return move, attack




class JUDGE():
    def __init__(self):
        self.replay_buffer = []
        self.memory_size = REPLAY_SIZE
        self.prio_max  = PRIO_MAX
        self.a = A
        self.e = E

        self.batch_size = BATCH_SIZE
        self.action_attack_size = ACTIONATTACK_SIZE
        self.action_move_size = ACTIONMOVE_SIZE
        self.all_blood = 0
        self.boss_all_blood = BOSS_ALL_BLOOD
        self.boss_blood = BOSS_ALL_BLOOD
        # self.all_power = 0
        self.choose_action_time = CHOOSE_ACTION_TIME  # 根据时间指导随机的操作选择引导
        self.all_loss = torch.tensor(0.)
        self.position = 0

    # 经验回放
    # def train_network(self, policy_net, target_net, policymove_net, targetmove_net, batch_train_size, flag, done, num_step):
    def train_network(self, policy_net, batch_train_size, flag, done, num_step):
        # step 1: obtain random minibatch from replay memory!
        dataInx = random.randint(0, len(self.replay_buffer) - 1)

        station_batch = self.replay_buffer[dataInx][0]
        attack_action_batch = self.replay_buffer[dataInx][1].argmax()
        move_action_batch = self.replay_buffer[dataInx][2].argmax()
        reward_batch = self.replay_buffer[dataInx][3]
        next_station_batch = self.replay_buffer[dataInx][4]

        for i in range(1, batch_train_size):
            dataInx = random.randint(0, len(self.replay_buffer) - 1)

            station_batch = np.vstack([station_batch, self.replay_buffer[dataInx][0]])
            attack_action_batch = np.vstack([attack_action_batch, self.replay_buffer[dataInx][1].argmax()])
            move_action_batch = np.vstack([move_action_batch, self.replay_buffer[dataInx][2].argmax()])
            reward_batch = np.vstack([reward_batch, self.replay_buffer[dataInx][3]])
            next_station_batch = np.vstack([next_station_batch, self.replay_buffer[dataInx][4]])


        attack_action_batch = torch.as_tensor(attack_action_batch, dtype=torch.int64).to(DEVICE)
        move_action_batch = torch.as_tensor(move_action_batch, dtype=torch.int64).to(DEVICE)
        reward_batch = torch.as_tensor(reward_batch, dtype=torch.float32).to(DEVICE)
        # print("[+] station_batch", np.array(station_batch).shape)
        # print("[+] action_batch", action_batch)
        # print("[+] reward_batch", reward_batch)
        # print("[+] next_station_batch", np.array(next_station_batch).shape)

        move, attack = policy_net(station_batch)
        move_q = move.gather(1, move_action_batch)
        attack_q = attack.gather(1, attack_action_batch)
        tq = reward_batch

        # 优化Cueling
        # attack_current_q = policy_net(next_station_batch, batch_train_size).argmax(1).reshape(-1, 1)
        # attack_q_next = target_net(next_station_batch, batch_train_size).detach().gather(1, attack_current_q)
        # attack_tq = reward_batch + GAMMA * attack_q_next

        # 取最大值
        # q_next = target_net(next_station_batch, batch_train_size).detach().max(1)[0].reshape(-1, 1)
        # tq = reward_batch + GAMMA * q_next

        # move_q = policymove_net(station_batch, batch_train_size).gather(1, move_action_batch)

        # 优化Cueling
        # move_current_q = policymove_net(next_station_batch, batch_train_size).argmax(1).reshape(-1, 1)
        # move_q_next = targetmove_net(next_station_batch, batch_train_size).detach().gather(1, move_current_q)
        # move_tq = reward_batch + GAMMA * move_q_next
        # move_tq = reward_batch

        # q_next = targetmove_net(next_station_batch, batch_train_size).detach().max(1)[0].reshape(-1, 1)
        # # print("q_next", q_next)
        # # print("q_next", q_next)
        # tq = reward_batch + GAMMA * q_next
        # print("reward_batch", reward_batch)
        # print("tq", tq)

        if done == 1:
            print("[*] move_q_predict:", move_q)
            # print("[*] move_tq_predict:", move_tq)
            print("[*] attack_q_predict:", attack_q)
            # print("[*] attack_tq_predict:", attack_tq)


        # print(q_next)
        # print("tq", tq)

        # loss = policy_net.mls(q, tq).requires_grad_(True)
        move_loss = policy_net.mls(move_q, tq)
        attack_loss = policy_net.mls(attack_q, tq)


        self.all_loss += move_loss + attack_loss


        policy_net.opt.zero_grad()
        move_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
        policy_net.opt.step()

        policy_net.opt.zero_grad()
        attack_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
        policy_net.opt.step()


    def choose_action(self, policy_net, station, training_num):
        mutetu = (training_num / TOTAL_NUM)
        if mutetu >= 1.0:
            mutetu = 0.9
        if random.random() >= mutetu:
            return random.randint(0, self.action_attack_size-1), random.randint(0, self.action_move_size-1), "   随机"+str(mutetu)
        else:
            move, attack = policy_net(station)
            move = move.detach().cpu()
            attack = attack.detach().cpu()
            # move_state = policymove_net(station, 1).detach().cpu()
            # print("[*] choose state:", state)
            # w = heapq.nlargest(10, range(len(a)), a.take)
            return np.argmax(move), np.argmax(attack), "<- 网络"+str(mutetu)



    # 进行reward
    # 由于无法观测到boss血量，使用时间作为reward, reward为持续时间 tqt
    # 自身费血将会受到-reward
    def action_judge(self, next_self_blood, next_boss_blood, attack_action, move_action, pass_attack_action, pass_move_action, self_power):
        # 持续时间为reward  费血 -100 回血 +100 攻击 +0.5 增加能量 +10
        # reward在1附近
        # print(self.boss_blood)
        return_reward = 0
        if next_boss_blood <= 20 and self.boss_blood <= 160: # boss死亡
            reward = 50
            done = 1
            return reward, done, self.boss_blood

        if next_self_blood < 1:  # 死亡
            reward = -45
            done = 1
            return reward, done, self.boss_blood


        done = 0

        if self.all_blood - next_self_blood >= 1:  # 费血
            return_reward += -26 * (self.all_blood - next_self_blood)
            self.all_blood = next_self_blood
        # elif next_self_blood - self.all_blood >= 1 and attack_action == Q_LONG_USE:  # 回血
        #     return_reward += 25 * (next_self_blood - self.all_blood)
        #     self.all_blood = next_self_blood


        # if action == I_ATTACK or action == J_LEFT or action == K_ATTACK or action == L_RIGHT or action == ATTACK_NUM:  # 攻击给予正反馈
        #     if emergence_break < 2:
        #         return_reward += 8
        #     else:
        #         return_reward += 8
        #         emergence_break = 100

        # if next_self_power - self.all_power >= 5 and (attack_action == I_ATTACK or attack_action == K_ATTACK or attack_action == ATTACK_NUM or attack_action == ATTACK_DOUBLE_NUM):  # 增加能量
        #     return_reward += 25
        #     self.all_power = next_self_power

        # if self.boss_blood - next_boss_blood >= 8 and (attack_action == I_ATTACK or attack_action == K_ATTACK or attack_action == ATTACK_NUM
        #      or attack_action == ATTACK_DOUBLE_NUM or attack_action == Q_SHORT_USE or attack_action == IQ_BOOM or attack_action == KQ_BOOM
        #      or move_action == E_RUN):  # 增加能量
        if self.boss_blood - next_boss_blood >= 8:  # boss血量减少
            return_reward += (self.boss_blood - next_boss_blood) / 4
            self.boss_blood = next_boss_blood
            # print(self.boss_blood)

        if self_power <= 18 and (attack_action == IQ_BOOM or attack_action == KQ_BOOM or attack_action == Q_SHORT_USE):
            return_reward -= 2


        # if self.all_power - next_self_power >= 4:
        #     self.all_power = next_self_power

        # if return_reward == 0.0 and pass_move_action == move_action:
        #     return_reward -= 3

        # if emergence_break < 2:
        #     return_reward += time.time() - init_time
        #     init_time = time.time()
        # else:
        #     return_reward += time.time() - init_time
        #     init_time = time.time()
        #     emergence_break = 100
        # if -0.01 <= return_reward <= 0.01:
        #     return_reward += -0.02

        # if return_reward >= 0.0 and (action == J_LEFT or action == L_RIGHT or action == 5 or action == 6 or action == 7):
        #     return_reward += 4

        # if return_reward == 0.0 and (action == I_ATTACK or action == K_ATTACK or action == ATTACK_NUM):
        #     return_reward -= 1

        return return_reward, done, self.boss_blood


    def store_data(self, station, attack_action, move_action, reward, next_station):
        one_hot_attackaction = np.zeros(self.action_attack_size)
        one_hot_attackaction[attack_action] = 1
        one_hot_moveaction = np.zeros(self.action_move_size)
        one_hot_moveaction[move_action] = 1

        # print("[*] REPLAY", self.replay_buffer[0])
        if len(self.replay_buffer) >= REPLAY_SIZE:
            # print("first:", self.replay_buffer[self.position % REPLAY_SIZE])
            self.replay_buffer[self.position % REPLAY_SIZE] = (station, one_hot_attackaction, one_hot_moveaction, reward, next_station)
            # print("second:", self.replay_buffer[self.position % REPLAY_SIZE])
            self.position += 1
        else:
            self.replay_buffer.append([station, one_hot_attackaction, one_hot_moveaction, reward, next_station])

