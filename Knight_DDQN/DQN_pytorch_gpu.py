import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import os
import random
import numpy as np
import pickle

from argconfig import *


'''
===================================== Move model
'''
class MoveResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(MoveResBlk, self).__init__()
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


class MoveDQN(nn.Module):
    def __init__(self, move_action):
        super(MoveDQN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(3, 8, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        self.blk1 = MoveResBlk(8, 16, stride=2)
        self.blk2 = MoveResBlk(16, 32, stride=2)
        self.blk3 = MoveResBlk(32, 64, stride=2)
        self.blk4 = MoveResBlk(64, 64, stride=2)

        # linear layer 线性层
        # 线性层的输入取决于conv2d的输出，计算输入图像大小
        self.linear_input_size = 3840

        self.advantage = nn.Sequential(  # 判定机制
            nn.Linear(self.linear_input_size, 256),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1),
        )

        self.move = nn.Sequential(
            nn.Linear(self.linear_input_size, 256),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(128, move_action),
        )

        self.mls = nn.MSELoss()
        self.relu = nn.ReLU(inplace=True)
        self.opt = torch.optim.Adam(self.parameters(), lr=LR_MOVE)

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32).to(DEVICE)

        x = self.conv1(x)
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        x = torch.flatten(x, 1)  # 打平

        advantage = self.advantage(x)
        move = self.move(x)

        return move + advantage - move.mean()

'''
===================================== Attack model
'''
class AttackResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(AttackResBlk, self).__init__()
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

class AttackDQN(nn.Module):
    def __init__(self, attack_action):
        super(AttackDQN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(3, 8, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        self.blk1 = AttackResBlk(8, 16, stride=2)
        self.blk2 = AttackResBlk(16, 32, stride=2)
        self.blk3 = AttackResBlk(32, 64, stride=2)
        self.blk4 = AttackResBlk(64, 64, stride=2)

        # linear layer 线性层
        # 线性层的输入取决于conv2d的输出，计算输入图像大小
        self.linear_input_size = 3840

        self.advantage = nn.Sequential(  # 判定机制
            nn.Linear(self.linear_input_size, 256),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1),
        )

        self.attack = nn.Sequential(
            nn.Linear(self.linear_input_size, 256),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(128, attack_action),
        )

        self.mls = nn.MSELoss()
        self.relu = nn.ReLU(inplace=True)
        self.opt = torch.optim.Adam(self.parameters(), lr=LR_ATTACK)

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32).to(DEVICE)

        x = self.conv1(x)
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        x = torch.flatten(x, 1)

        advantage = self.advantage(x)
        attack = self.attack(x)

        return attack + advantage - attack.mean()

# '''
# ===================================== ICM model
# '''
# class ICM(nn.Module):
#     def __init__(self, in_channels=3, num_actions=5):
#         """
#         Initialize a deep Q-learning network as described in
#         https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
#         Arguments:
#             in_channels: number of channel of input.
#                 i.e The number of most recent frames stacked together as describe in the paper
#             num_actions: number of action-value to output, one-to-one correspondence to action in game.
#         """
#         super(ICM, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
#         self.fc4 = nn.Linear(7 * 7 * 64, 512)
#
#         self.pred_module1 = nn.Linear(512 + num_actions, 256)
#         self.pred_module2 = nn.Linear(256, 512)
#
#         self.invpred_module1 = nn.Linear(512 + 512, 256)
#         self.invpred_module2 = nn.Linear(256, num_actions)
#
#     def get_feature(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.fc4(x.view(x.size(0), -1)))
#         return x
#
#     def forward(self, x):
#         # get feature
#         feature_x = self.get_feature(x)
#         return feature_x
#
#     def get_full(self, x, x_next, a_vec):
#         # get feature
#         feature_x = self.get_feature(x)
#         feature_x_next = self.get_feature(x_next)
#
#         pred_s_next = self.pred(feature_x, a_vec)  # predict next state feature
#         pred_a_vec = self.invpred(feature_x, feature_x_next)  # (inverse) predict action
#
#         return pred_s_next, pred_a_vec, feature_x_next
#
#     def pred(self, feature_x, a_vec):
#         # Forward prediction: predict next state feature, given current state feature and action (one-hot)
#         pred_s_next = F.relu(self.pred_module1(torch.cat([feature_x, a_vec.float()], dim=-1).detach()))
#         pred_s_next = self.pred_module2(pred_s_next)
#         return pred_s_next
#
#     def invpred(self, feature_x, feature_x_next):
#         # Inverse prediction: predict action (one-hot), given current and next state features
#         pred_a_vec = F.relu(self.invpred_module1(torch.cat([feature_x, feature_x_next], dim=-1)))
#         pred_a_vec = self.invpred_module2(pred_a_vec)
#         return F.softmax(pred_a_vec, dim=-1)


'''
=======================================回放经验池
'''
class Replay_buffer():
    def __init__(self, max_size=REPLAY_SIZE, max_possize=REPLAY_POSSIZE):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

        self.posstorage = []
        self.max_possize = max_possize
        self.posptr = 0


    def push(self, station, move_action, attack_action, reward, next_station):
        one_hot_moveaction = np.zeros(ACTIONMOVE_SIZE)
        one_hot_moveaction[move_action] = 1
        one_hot_attackaction = np.zeros(ACTIONATTACK_SIZE)
        one_hot_attackaction[attack_action] = 1


        data = [station, one_hot_moveaction, one_hot_attackaction, reward, next_station]

        if reward > 0:
            if len(self.posstorage) == self.max_possize:
                self.posstorage[self.posptr % self.max_possize] = data
                self.posptr = self.posptr % self.max_possize + 1
            else:
                self.posstorage.append(data)
                self.posptr += 1

        if len(self.storage) == self.max_size:
            self.storage[self.ptr % self.max_size] = data
            self.ptr = self.ptr % self.max_size + 1
        else:
            self.storage.append(data)
            self.ptr += 1

    def get(self, batch_size):
        dataInx = random.randint(0, len(self.storage) - 1)

        station_batch = self.storage[dataInx][0]
        move_action_batch = self.storage[dataInx][1].argmax()
        attack_action_batch = self.storage[dataInx][2].argmax()
        reward_batch = self.storage[dataInx][3]
        next_station_batch = self.storage[dataInx][4]

        for i in range(1, batch_size):
            dataInx = random.randint(0, len(self.storage) - 1)

            station_batch = np.vstack([station_batch, self.storage[dataInx][0]])
            move_action_batch = np.vstack([move_action_batch, self.storage[dataInx][1].argmax()])
            attack_action_batch = np.vstack([attack_action_batch, self.storage[dataInx][2].argmax()])
            reward_batch = np.vstack([reward_batch, self.storage[dataInx][3]])
            next_station_batch = np.vstack([next_station_batch, self.storage[dataInx][4]])

        move_action_batch = torch.as_tensor(move_action_batch, dtype=torch.int64).to(DEVICE)
        attack_action_batch = torch.as_tensor(attack_action_batch, dtype=torch.int64).to(DEVICE)
        reward_batch = torch.as_tensor(reward_batch, dtype=torch.float32).to(DEVICE)

        return station_batch, move_action_batch, attack_action_batch, reward_batch, next_station_batch

    def getpos(self, batch_size):
        dataInx = random.randint(0, len(self.posstorage) - 1)

        station_batch = self.posstorage[dataInx][0]
        move_action_batch = self.posstorage[dataInx][1].argmax()
        attack_action_batch = self.posstorage[dataInx][2].argmax()
        reward_batch = self.posstorage[dataInx][3]
        next_station_batch = self.posstorage[dataInx][4]

        for i in range(1, batch_size):
            dataInx = random.randint(0, len(self.posstorage) - 1)

            station_batch = np.vstack([station_batch, self.posstorage[dataInx][0]])
            move_action_batch = np.vstack([move_action_batch, self.posstorage[dataInx][1].argmax()])
            attack_action_batch = np.vstack([attack_action_batch, self.posstorage[dataInx][2].argmax()])
            reward_batch = np.vstack([reward_batch, self.posstorage[dataInx][3]])
            next_station_batch = np.vstack([next_station_batch, self.posstorage[dataInx][4]])

        move_action_batch = torch.as_tensor(move_action_batch, dtype=torch.int64).to(DEVICE)
        attack_action_batch = torch.as_tensor(attack_action_batch, dtype=torch.int64).to(DEVICE)
        reward_batch = torch.as_tensor(reward_batch, dtype=torch.float32).to(DEVICE)

        return station_batch, move_action_batch, attack_action_batch, reward_batch, next_station_batch

    def save(self):
        print("[-] Replay_buffer saving...")
        if os.path.exists(DQN_STORE_PATH):
            os.remove(DQN_STORE_PATH)
        pickle.dump(self.storage, open(DQN_STORE_PATH, 'wb'))
        if os.path.exists(DQN_POSSTORE_PATH):
            os.remove(DQN_POSSTORE_PATH)
        pickle.dump(self.posstorage, open(DQN_POSSTORE_PATH, 'wb'))
        print("[+] Replay_buffer finish save!")


    def load(self):
        if os.path.exists(DQN_STORE_PATH):
            self.storage = pickle.load(open(DQN_STORE_PATH, 'rb'))
            print("[*] REPLAY_BUFFER load finish! len:", len(self.storage))
        if os.path.exists(DQN_POSSTORE_PATH):
            self.posstorage = pickle.load(open(DQN_POSSTORE_PATH, 'rb'))
            print("[*] REPLAY_BUFFER load finish! len:", len(self.posstorage))


'''
==========================================DDQN model
'''
class DDQN():
    def __init__(self):
        self.batch_size = BATCH_SIZE
        self.action_move_size = ACTIONMOVE_SIZE
        self.action_attack_size = ACTIONATTACK_SIZE
        self.all_blood = 0
        self.boss_all_blood = BOSS_ALL_BLOOD
        self.boss_blood = BOSS_ALL_BLOOD
        # self.all_power = 0
        self.all_loss = torch.tensor(0.)
        self.pass_count = 0

        self.MoveDQN_eval = MoveDQN(ACTIONMOVE_SIZE).to(DEVICE)
        self.MoveDQN_target = MoveDQN(ACTIONMOVE_SIZE).to(DEVICE)

        self.AttackDQN_eval = AttackDQN(ACTIONATTACK_SIZE).to(DEVICE)
        self.AttackDQN_target = AttackDQN(ACTIONATTACK_SIZE).to(DEVICE)


        if not os.path.exists("model"):
            os.mkdir("model")

    def choose_action(self, station, training_num):
        mutetu = (training_num / TOTAL_NUM)
        if mutetu >= 1.0:
            mutetu = 0.9
        if random.random() >= mutetu:
            return random.randint(0, self.action_move_size-1), random.randint(0, self.action_attack_size-1), "   随机"+str(mutetu), False
        else:
            move = self.MoveDQN_eval(station)
            attack = self.AttackDQN_eval(station)

            move = np.array(move.detach().cpu()[0])
            attack = np.array(attack.detach().cpu()[0])
            return move, attack, "<- 网络"+str(mutetu), True

    # 经验回放
    def train_network(self, replay_buffer, batch_train_size, done, flag, num_step):
        # 软更新网络参数
        for target_param, param in zip(self.MoveDQN_target.parameters(), self.MoveDQN_eval.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

        for target_param, param in zip(self.AttackDQN_target.parameters(), self.AttackDQN_eval.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

        # step 1: 从 replay memory 随机采样 # TODO 线程sumtree!
        if flag == 0:
            station_batch, move_action_batch, attack_action_batch, reward_batch, next_station_batch = replay_buffer.get(batch_train_size)
        elif flag == 1:  # 多回放正reward的案例(好操作)
            station_batch, move_action_batch, attack_action_batch, reward_batch, next_station_batch = replay_buffer.getpos(batch_train_size)


        # ===================================Move动作网络
        move_q = self.MoveDQN_eval(station_batch).gather(1, move_action_batch)
        # 优化Cueling
        move_current_q = self.MoveDQN_eval(next_station_batch).argmax(1).reshape(-1, 1)
        move_q_next = self.MoveDQN_target(next_station_batch).detach().gather(1, move_current_q)
        move_tq = reward_batch + GAMMA * move_q_next

        move_loss = self.MoveDQN_eval.mls(move_q, move_tq)

        self.MoveDQN_eval.opt.zero_grad()
        move_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.MoveDQN_eval.parameters(), 10)
        self.MoveDQN_eval.opt.step()

        # ===================================Attack动作网络
        attack_q = self.AttackDQN_eval(station_batch).gather(1, attack_action_batch)
        # 优化Cueling
        attack_current_q = self.AttackDQN_eval(next_station_batch).argmax(1).reshape(-1, 1)
        attack_q_next = self.AttackDQN_target(next_station_batch).detach().gather(1, attack_current_q)
        attack_tq = reward_batch + GAMMA * attack_q_next

        attack_loss = self.AttackDQN_eval.mls(attack_q, attack_tq)

        self.AttackDQN_eval.opt.zero_grad()
        attack_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.AttackDQN_eval.parameters(), 10)
        self.AttackDQN_eval.opt.step()


        if done:
            print("[*] move_q_predict:", move_q)
            print("[*] move_tq_predict:", move_tq)
            print("[*] attack_q_predict:", attack_q)
            print("[*] attack_tq_predict:", attack_tq)

        self.all_loss += move_loss + attack_loss

    # 进行reward, 由于无法观测到boss血量，使用时间作为reward, reward为持续时间 自身费血将会受到-reward
    # def action_judge(self, next_self_blood, next_boss_blood, attack_action, move_action, pass_attack_action, pass_move_action, self_power):
    def action_judge(self, next_self_blood, next_boss_blood, move_num, attack_num, last_move_num, self_power, done_is):
        # 持续时间为reward  费血 -26 回血 +26 攻击 +0.5 增加能量 +10
        # reward在1附近 # TODO
        return_reward = 0
        if next_boss_blood <= 20 and self.boss_blood <= 160:  # boss死亡
            reward = 25
            done = 1
            self.pass_count += 1
            return reward, done, self.boss_blood

        if next_self_blood < 1:  # 死亡
            if done_is:
                reward = -22
            else:
                reward = -1.75
            done = 1
            return reward, done, self.boss_blood

        done = 0

        if self.all_blood - next_self_blood >= 1:  # 费血
            return_reward += -13 * (self.all_blood - next_self_blood)
            self.all_blood = next_self_blood

        if self.boss_blood - next_boss_blood >= 8:  # boss血量减少
            return_reward += (self.boss_blood - next_boss_blood) / 8
            self.boss_blood = next_boss_blood

        if self_power <= 18 and (
                attack_num == IQ_BOOM or attack_num == KQ_BOOM or attack_num == Q_SHORT_USE):  # 没能量放技能惩罚
            return_reward -= 1

        if return_reward == 0:  # 未击中惩罚
            return_reward -= 0.25

        if move_num == last_move_num and return_reward <= 0:  # 相同移动惩罚
            return_reward -= 0.5

        return return_reward, done, self.boss_blood

    def save(self, label=None):
        print("[-] Model saving...")
        if label==None:
            torch.save(self.MoveDQN_eval.state_dict(), DQN_MODEL_PATH+str("MoveDQN_eval.pth"))
            torch.save(self.MoveDQN_target.state_dict(), DQN_MODEL_PATH+str("MoveDQN_target.pth"))
            torch.save(self.AttackDQN_eval.state_dict(), DQN_MODEL_PATH+str("AttackDQN_eval.pth"))
            torch.save(self.AttackDQN_target.state_dict(), DQN_MODEL_PATH+str("AttackDQN_target.pth"))
        else:
            label += "_"+str(time.strftime("%Y%m%d%H%M%S", time.localtime()))
            print("[-] Model "+str(label)+"saving...")
            torch.save(self.MoveDQN_eval.state_dict(), DQN_MODEL_PATH + str("MoveDQN_eval"+str(label)+".pth"))
            torch.save(self.MoveDQN_target.state_dict(), DQN_MODEL_PATH + str("MoveDQN_target"+str(label)+".pth"))
            torch.save(self.AttackDQN_eval.state_dict(), DQN_MODEL_PATH + str("AttackDQN_eval"+str(label)+".pth"))
            torch.save(self.AttackDQN_target.state_dict(), DQN_MODEL_PATH + str("AttackDQN_target"+str(label)+".pth"))
        print("[+] Model finish save!")

    def load(self):
        if os.path.exists(DQN_MODEL_PATH+str("MoveDQN_eval.pth")) and os.path.exists(DQN_MODEL_PATH+str("MoveDQN_target.pth")) \
                and os.path.exists(DQN_MODEL_PATH+str("AttackDQN_eval.pth")) and os.path.exists(DQN_MODEL_PATH+str("AttackDQN_target.pth")):
            self.MoveDQN_eval.load_state_dict(torch.load(DQN_MODEL_PATH+str("MoveDQN_eval.pth")))
            self.MoveDQN_target.load_state_dict(torch.load(DQN_MODEL_PATH+str("MoveDQN_target.pth")))
            self.AttackDQN_eval.load_state_dict(torch.load(DQN_MODEL_PATH+str("AttackDQN_eval.pth")))
            self.AttackDQN_target.load_state_dict(torch.load(DQN_MODEL_PATH+str("AttackDQN_target.pth")))
            print("[*] model load finish!")
