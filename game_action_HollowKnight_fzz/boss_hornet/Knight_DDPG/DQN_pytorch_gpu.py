import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Categorical

import os
import random
import numpy as np
import pickle
from argconfig import *



'''
===================================== Move Actor_Critic
'''
class MoveActorResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(MoveActorResBlk, self).__init__()
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

class MoveActor(nn.Module):
    def __init__(self, move_action):
        super(MoveActor, self).__init__()
        self.mls = nn.MSELoss()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(3, 8, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        self.blk1 = MoveActorResBlk(8, 16, stride=2)
        self.blk2 = MoveActorResBlk(16, 32, stride=2)
        self.blk3 = MoveActorResBlk(32, 64, stride=2)
        self.blk4 = MoveActorResBlk(64, 64, stride=2)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # linear layer 线性层
        # 线性层的输入取决于conv2d的输出，计算输入图像大小
        self.linear_input_size = 3840

        self.move = nn.Sequential(
            nn.Linear(self.linear_input_size, 256),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Dropout(p=0.8),
            nn.Linear(128, move_action),
        )




    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32).to(DEVICE)
        # print(x.shape)

        x = self.conv1(x)
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        # x = self.conv(x)

        x = torch.flatten(x, 1) # 打平

        move = self.move(x)

        return move


class MoveCriticResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(MoveCriticResBlk, self).__init__()
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

class MoveCritic(nn.Module):

    def __init__(self):
        super(MoveCritic, self).__init__()
        self.mls = nn.MSELoss()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(3, 8, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        self.blk1 = MoveCriticResBlk(8, 16, stride=2)
        self.blk2 = MoveCriticResBlk(16, 32, stride=2)
        self.blk3 = MoveCriticResBlk(32, 64, stride=2)
        self.blk4 = MoveCriticResBlk(64, 64, stride=2)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # linear layer 线性层
        # 线性层的输入取决于conv2d的输出，计算输入图像大小
        self.linear_input_size = 3840

        self.move_state = nn.Sequential(
            nn.Linear(self.linear_input_size, 256),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),
        )
        self.move_act = nn.Linear(ACTIONMOVE_SIZE, 64)
        self.move_out = nn.Linear(128, 1)



    def forward(self, state, move_batch):
        state = torch.as_tensor(state, dtype=torch.float32).to(DEVICE)
        # print(x.shape)

        state = self.conv1(state)
        state = self.blk1(state)
        state = self.blk2(state)
        state = self.blk3(state)
        state = self.blk4(state)
        # x = self.conv(x)

        state = torch.flatten(state, 1)  # 打平

        move_stat = self.move_state(state)
        move_ac = self.move_act(move_batch)
        move_net = torch.tanh(torch.cat([move_stat, move_ac], 1))
        move = self.move_out(move_net)

        return move



'''
===================================== Attack Actor_Critic
'''
class AttackActorResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(AttackActorResBlk, self).__init__()
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

class AttackActor(nn.Module):
    def __init__(self, attack_action):
        super(AttackActor, self).__init__()
        self.mls = nn.MSELoss()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(3, 8, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        self.blk1 = AttackActorResBlk(8, 16, stride=2)
        self.blk2 = AttackActorResBlk(16, 32, stride=2)
        self.blk3 = AttackActorResBlk(32, 64, stride=2)
        self.blk4 = AttackActorResBlk(64, 64, stride=2)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # linear layer 线性层
        # 线性层的输入取决于conv2d的输出，计算输入图像大小
        self.linear_input_size = 3840

        self.attack = nn.Sequential(
            nn.Linear(self.linear_input_size, 256),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Dropout(p=0.8),
            nn.Linear(128, attack_action),
        )


    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32).to(DEVICE)
        # print(x.shape)

        x = self.conv1(x)
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        # x = self.conv(x)

        x = torch.flatten(x, 1) # 打平

        attack = self.attack(x)

        return attack


class AttackCriticResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(AttackCriticResBlk, self).__init__()
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

class AttackCritic(nn.Module):

    def __init__(self):
        super(AttackCritic, self).__init__()
        self.mls = nn.MSELoss()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(3, 8, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        self.blk1 = AttackCriticResBlk(8, 16, stride=2)
        self.blk2 = AttackCriticResBlk(16, 32, stride=2)
        self.blk3 = AttackCriticResBlk(32, 64, stride=2)
        self.blk4 = AttackCriticResBlk(64, 64, stride=2)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # linear layer 线性层
        # 线性层的输入取决于conv2d的输出，计算输入图像大小
        self.linear_input_size = 3840

        self.attack_state = nn.Sequential(
            nn.Linear(self.linear_input_size, 256),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),
        )
        self.attack_act = nn.Linear(ACTIONATTACK_SIZE, 64)
        self.attack_out = nn.Linear(128, 1)


    def forward(self, state, attack_batch):
        state = torch.as_tensor(state, dtype=torch.float32).to(DEVICE)
        # print(x.shape)

        state = self.conv1(state)
        state = self.blk1(state)
        state = self.blk2(state)
        state = self.blk3(state)
        state = self.blk4(state)
        # x = self.conv(x)

        state = torch.flatten(state, 1)  # 打平

        attack_stat = self.attack_state(state)
        attack_ac = self.attack_act(attack_batch)
        attack_net = torch.tanh(torch.cat([attack_stat, attack_ac], 1))
        attack = self.attack_out(attack_net)

        return attack

'''
=======================================回放经验池
'''
class Replay_buffer():
    def __init__(self, max_size=REPLAY_SIZE):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0


    def push(self, station, move_action, attack_action, reward, next_station):

        data = [station, move_action, attack_action, reward, next_station]

        if len(self.storage) == self.max_size:
            self.storage[self.ptr % self.max_size] = data
            self.ptr = self.ptr % self.max_size + 1
        else:
            self.storage.append(data)
            self.ptr += 1


    def get(self, batch_size):
        dataInx = random.randint(0, len(self.storage) - 1)

        station_batch = self.storage[dataInx][0]
        move_action_batch = self.storage[dataInx][1]
        attack_action_batch = self.storage[dataInx][2]
        reward_batch = self.storage[dataInx][3]
        next_station_batch = self.storage[dataInx][4]

        for i in range(1, batch_size):
            dataInx = random.randint(0, len(self.storage) - 1)

            station_batch = np.vstack([station_batch, self.storage[dataInx][0]])
            move_action_batch = np.vstack([move_action_batch, self.storage[dataInx][1]])
            attack_action_batch = np.vstack([attack_action_batch, self.storage[dataInx][2]])
            reward_batch = np.vstack([reward_batch, self.storage[dataInx][3]])
            next_station_batch = np.vstack([next_station_batch, self.storage[dataInx][4]])

        move_action_batch = torch.as_tensor(move_action_batch, dtype=torch.float32).to(DEVICE)
        attack_action_batch = torch.as_tensor(attack_action_batch, dtype=torch.float32).to(DEVICE)
        reward_batch = torch.as_tensor(reward_batch, dtype=torch.float32).to(DEVICE)

        return station_batch, move_action_batch, attack_action_batch, reward_batch, next_station_batch

    def save(self):
        print("[-] Replay_buffer saving...")
        if os.path.exists(DQN_STORE_PATH):
            os.remove(DQN_STORE_PATH)
        pickle.dump(self.storage, open(DQN_STORE_PATH, 'wb'))
        print("[+] Replay_buffer finish save!")


    def load(self):
        if os.path.exists(DQN_STORE_PATH):
            self.storage = pickle.load(open(DQN_STORE_PATH, 'rb'))
            print("[*] REPLAY_BUFFER load finish! len:", len(self.storage))

'''
==========================================DDPG model
'''
class DDPG():
    def __init__(self):
        self.batch_size = BATCH_SIZE
        self.all_blood = 0
        self.boss_all_blood = BOSS_ALL_BLOOD
        self.boss_blood = BOSS_ALL_BLOOD
        # self.all_power = 0
        self.all_loss = torch.tensor(0.)
        self.position = 0

        self.MoveActor_eval = MoveActor(ACTIONMOVE_SIZE).to(DEVICE)
        self.MoveActor_target = MoveActor(ACTIONMOVE_SIZE).to(DEVICE)
        self.MoveCritic_eval = MoveCritic().to(DEVICE)
        self.MoveCritic_target = MoveCritic().to(DEVICE)

        self.AttackActor_eval = AttackActor(ACTIONATTACK_SIZE).to(DEVICE)
        self.AttackActor_target = AttackActor(ACTIONATTACK_SIZE).to(DEVICE)
        self.AttackCritic_eval = AttackCritic().to(DEVICE)
        self.AttackCritic_target = AttackCritic().to(DEVICE)

        self.moveatrain = torch.optim.Adam(self.MoveActor_eval.parameters(), lr=LR_A)
        self.movectrain = torch.optim.Adam(self.MoveCritic_eval.parameters(), lr=LR_C)

        self.attackatrain = torch.optim.Adam(self.AttackActor_eval.parameters(), lr=LR_A)
        self.attackctrain = torch.optim.Adam(self.AttackCritic_eval.parameters(), lr=LR_C)

        self.move_losstd = nn.MSELoss()
        self.attack_losstd = nn.MSELoss()

        # for m in self.MoveActor_eval.modules(): # kaiming初始化
        #     if isinstance(m, (nn.Conv2d, nn.Linear)):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        # for m in self.MoveActor_target.modules():
        #     if isinstance(m, (nn.Conv2d, nn.Linear)):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        # for m in self.MoveCritic_eval.modules():
        #     if isinstance(m, (nn.Conv2d, nn.Linear)):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        # for m in self.MoveCritic_target.modules():
        #     if isinstance(m, (nn.Conv2d, nn.Linear)):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #
        # for m in self.AttackActor_eval.modules(): # kaiming初始化
        #     if isinstance(m, (nn.Conv2d, nn.Linear)):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        # for m in self.AttackActor_target.modules():
        #     if isinstance(m, (nn.Conv2d, nn.Linear)):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        # for m in self.AttackCritic_eval.modules():
        #     if isinstance(m, (nn.Conv2d, nn.Linear)):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        # for m in self.AttackCritic_target.modules():
        #     if isinstance(m, (nn.Conv2d, nn.Linear)):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        if not os.path.exists("model"):
            os.mkdir("model")

    def choose_action(self, station):
        move = self.MoveActor_eval(station)
        attack = self.AttackActor_eval(station)

        move = np.array(move.detach().cpu()[0])
        attack = np.array(attack.detach().cpu()[0])

        return move, attack


    # 经验回放
    def train_network(self, replay_buffer, batch_train_size, flag, done, num_step):

        # 软更新网络参数
        for target_param, param in zip(self.MoveActor_target.parameters(), self.MoveActor_eval.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)
        for target_param, param in zip(self.MoveCritic_target.parameters(), self.MoveCritic_eval.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

        for target_param, param in zip(self.AttackActor_target.parameters(), self.AttackActor_eval.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)
        for target_param, param in zip(self.AttackCritic_target.parameters(), self.AttackCritic_eval.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

        # step 1: 从 replay memory 随机采样 # TODO 线程sumtree!
        station_batch, move_action_batch, attack_action_batch, reward_batch, next_station_batch = replay_buffer.get(batch_train_size)

        # ===================================Move动作网络
        move_a = self.MoveActor_eval(station_batch)
        move_q = self.MoveCritic_eval(station_batch, move_a)  # loss=-q=-ce(s, ae(s))  更新ae ae(s)=a ae(s_)=a_
        loss_move = -torch.mean(move_q)  # 如果a是正确行为，那么它的Q更贴近0
        if flag:
            print("[*] move预测-评价:", move_a, move_q)

        self.moveatrain.zero_grad()
        loss_move.backward()
        # torch.nn.utils.clip_grad_norm_(self.Actor_eval.parameters(), MAX_FORWORD)
        self.moveatrain.step()

        # =====================================Move策略网络

        next_move_a = self.MoveActor_target(next_station_batch)  # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        next_move_q = self.MoveCritic_target(next_station_batch, next_move_a)  # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        move_q_predict_true = reward_batch + GAMMA * next_move_q  # q_target = 负的

        move_q_predict = self.MoveCritic_eval(station_batch, move_action_batch)
        move_mls_loss = self.move_losstd(move_q_predict_true, move_q_predict)  # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba) 更新ce ,但这个ae(s)是记忆中的ba，让ce得出的Q靠近Q_target,让评价更准确


        if flag:
            print("[*] move策略真实-预测:", move_q_predict_true, move_q_predict)

        self.movectrain.zero_grad()
        move_mls_loss.backward()
        self.movectrain.step()


        # ===================================Attack动作网络
        attack_a = self.AttackActor_eval(station_batch)
        attack_q = self.AttackCritic_eval(station_batch, attack_a)  # loss=-q=-ce(s, ae(s))  更新ae ae(s)=a ae(s_)=a_
        loss_attack = -torch.mean(attack_q)  # 如果a是正确行为，那么它的Q更贴近0
        if flag:
            print("[*] attack预测-评价:", attack_a, attack_q)

        self.attackatrain.zero_grad()
        loss_attack.backward()
        self.attackatrain.step()

        # =====================================Attack策略网络

        next_attack_a = self.AttackActor_target(next_station_batch)  # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        next_attack_q = self.AttackCritic_target(next_station_batch, next_attack_a)  # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        attack_q_predict_true = reward_batch + GAMMA * next_attack_q  # q_target = 负的

        attack_q_predict = self.AttackCritic_eval(station_batch, attack_action_batch)
        attack_mls_loss = self.attack_losstd(attack_q_predict_true, attack_q_predict)  # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba) 更新ce ,但这个ae(s)是记忆中的ba，让ce得出的Q靠近Q_target,让评价更准确

        if flag:
            print("[*] attack策略真实-预测:", attack_q_predict_true, attack_q_predict)

        self.attackctrain.zero_grad()
        attack_mls_loss.backward()
        self.attackctrain.step()


        self.all_loss += loss_move + loss_attack + move_mls_loss + attack_mls_loss

    # 进行reward, 由于无法观测到boss血量，使用时间作为reward, reward为持续时间 自身费血将会受到-reward
    # def action_judge(self, next_self_blood, next_boss_blood, attack_action, move_action, pass_attack_action, pass_move_action, self_power):
    def action_judge(self, next_self_blood, next_boss_blood, move_num, attack_num, self_power, last_move_num, done_is):
        # 持续时间为reward  费血 -26 回血 +26 攻击 +0.5 增加能量 +10
        # reward在1附近 # TODO
        return_reward = 0
        if next_boss_blood <= 20 and self.boss_blood <= 160: # boss死亡
            reward = 25
            done = 1
            return reward, done, self.boss_blood

        if next_self_blood < 1:  # 死亡
            if done_is:
                reward = -22
            else:
                reward = 0
            done = 1
            return reward, done, self.boss_blood


        done = 0

        if self.all_blood - next_self_blood >= 1:  # 费血
            return_reward += -13 * (self.all_blood - next_self_blood)
            self.all_blood = next_self_blood

        if self.boss_blood - next_boss_blood >= 8:  # boss血量减少
            return_reward += (self.boss_blood - next_boss_blood) / 8
            self.boss_blood = next_boss_blood

        if self_power <= 18 and (attack_num == IQ_BOOM or attack_num == KQ_BOOM or attack_num == Q_SHORT_USE):  # 没能量放技能惩罚
            return_reward -= 0.5

        if return_reward == 0:  # 未击中惩罚
            return_reward -= 0.25

        if move_num == last_move_num and return_reward <= 0: # 相同移动惩罚
            return_reward -= 1

        return return_reward, done, self.boss_blood

    def save(self, label=None):
        print("[-] Model saving...")
        if label==None:
            torch.save(self.MoveActor_eval.state_dict(), DQN_MODEL_PATH+str("MoveActor_eval.pth"))
            torch.save(self.MoveActor_target.state_dict(), DQN_MODEL_PATH+str("MoveActor_target.pth"))
            torch.save(self.MoveCritic_eval.state_dict(), DQN_MODEL_PATH+str("MoveCritic_eval.pth"))
            torch.save(self.MoveCritic_target.state_dict(), DQN_MODEL_PATH+str("MoveCritic_target.pth"))

            torch.save(self.AttackActor_eval.state_dict(), DQN_MODEL_PATH + str("AttackActor_eval.pth"))
            torch.save(self.AttackActor_target.state_dict(), DQN_MODEL_PATH + str("AttackActor_target.pth"))
            torch.save(self.AttackCritic_eval.state_dict(), DQN_MODEL_PATH + str("AttackCritic_eval.pth"))
            torch.save(self.AttackCritic_target.state_dict(), DQN_MODEL_PATH + str("AttackCritic_target.pth"))
        else:
            label += "_"+str(time.strftime("%Y%m%d%H%M%S", time.localtime()))
            print("[-] Model "+str(label)+"saving...")
            torch.save(self.MoveActor_eval.state_dict(), DQN_MODEL_PATH + str("MoveActor_eval"+str(label)+".pth"))
            torch.save(self.MoveActor_target.state_dict(), DQN_MODEL_PATH + str("MoveActor_target"+str(label)+".pth"))
            torch.save(self.MoveCritic_eval.state_dict(), DQN_MODEL_PATH + str("MoveCritic_eval"+str(label)+".pth"))
            torch.save(self.MoveCritic_target.state_dict(), DQN_MODEL_PATH + str("MoveCritic_target"+str(label)+".pth"))

            torch.save(self.AttackActor_eval.state_dict(), DQN_MODEL_PATH + str("AttackActor_eval" + str(label) + ".pth"))
            torch.save(self.AttackActor_target.state_dict(), DQN_MODEL_PATH + str("AttackActor_target" + str(label) + ".pth"))
            torch.save(self.AttackCritic_eval.state_dict(), DQN_MODEL_PATH + str("AttackCritic_eval" + str(label) + ".pth"))
            torch.save(self.AttackCritic_target.state_dict(), DQN_MODEL_PATH + str("AttackCritic_target" + str(label) + ".pth"))
        print("[+] Model finish save!")

    def load(self):
        if os.path.exists(DQN_MODEL_PATH+str("MoveActor_eval.pth")) and os.path.exists(DQN_MODEL_PATH+str("MoveActor_target.pth")) \
                and os.path.exists(DQN_MODEL_PATH+str("MoveCritic_eval.pth")) and os.path.exists(DQN_MODEL_PATH+str("MoveCritic_target.pth")) \
                and os.path.exists(DQN_MODEL_PATH+str("AttackActor_eval.pth")) and os.path.exists(DQN_MODEL_PATH+str("AttackActor_target.pth")) \
                and os.path.exists(DQN_MODEL_PATH+str("AttackCritic_eval.pth")) and os.path.exists(DQN_MODEL_PATH+str("AttackCritic_target.pth")):
            self.MoveActor_eval.load_state_dict(torch.load(DQN_MODEL_PATH+str("MoveActor_eval.pth")))
            self.MoveActor_target.load_state_dict(torch.load(DQN_MODEL_PATH+str("MoveActor_target.pth")))
            self.MoveCritic_eval.load_state_dict(torch.load(DQN_MODEL_PATH+str("MoveCritic_eval.pth")))
            self.MoveCritic_target.load_state_dict(torch.load(DQN_MODEL_PATH+str("MoveCritic_target.pth")))

            self.AttackActor_eval.load_state_dict(torch.load(DQN_MODEL_PATH + str("AttackActor_eval.pth")))
            self.AttackActor_target.load_state_dict(torch.load(DQN_MODEL_PATH + str("AttackActor_target.pth")))
            self.AttackCritic_eval.load_state_dict(torch.load(DQN_MODEL_PATH + str("AttackCritic_eval.pth")))
            self.AttackCritic_target.load_state_dict(torch.load(DQN_MODEL_PATH + str("AttackCritic_target.pth")))
            print("[*] model load finish!")


