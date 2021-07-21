import os
import random
import numpy as np
import pickle
import torch

from config import *


# 回放经验池
class AmemoryReplaybuffer():
    def __init__(self, max_size=REPLAY_SIZE, positive_mode=POSITIVEMODE, max_possize=REPLAY_POSSIZE):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        self.posmode = positive_mode

        self.posstorage = []
        self.max_possize = max_possize
        self.posptr = 0

        # 字典存储对应一轮游戏的score
        self.rewardflag_dict = {}


    def push(self, state, action, reward, next_state):
        data = [state, action, reward, next_state]

        if reward > 0 and self.posmode:
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

        state_batch = self.storage[dataInx][0]
        action_batch = self.storage[dataInx][1]
        if self.storage[dataInx][2] > 100000:
            reward_batch = self.rewardflag_dict[self.storage[dataInx][2]]
        else:
            reward_batch = self.storage[dataInx][2]
        next_state_batch = self.storage[dataInx][3]

        for i in range(1, batch_size):
            dataInx = random.randint(0, len(self.storage) - 1)

            state_batch = torch.cat((state_batch, self.storage[dataInx][0]), 0)
            action_batch = np.vstack([action_batch, self.storage[dataInx][1]])
            if self.storage[dataInx][2] > 100000:
                reward_batch = np.vstack([reward_batch, self.rewardflag_dict[self.storage[dataInx][2]]])
            else:
                reward_batch = np.vstack([reward_batch, self.storage[dataInx][2]])
            next_state_batch = torch.cat((next_state_batch, self.storage[dataInx][3]), 0)

        action_batch = torch.as_tensor(action_batch, dtype=torch.long).to(DEVICE)
        reward_batch = torch.as_tensor(reward_batch, dtype=torch.float32).to(DEVICE)

        return state_batch, action_batch, reward_batch, next_state_batch

    def getpos(self, batch_size):
        dataInx = random.randint(0, len(self.posstorage) - 1)

        state_batch = self.posstorage[dataInx][0]
        action_batch = self.posstorage[dataInx][1]
        reward_batch = self.rewardflag_dict[self.posstorage[dataInx][2]]
        next_state_batch = self.posstorage[dataInx][3]

        for i in range(1, batch_size):
            dataInx = random.randint(0, len(self.posstorage) - 1)

            state_batch = torch.cat((state_batch, self.posstorage[dataInx][0]), 0)
            action_batch = np.vstack([action_batch, self.posstorage[dataInx][1]])
            reward_batch = np.vstack([reward_batch, self.rewardflag_dict[self.posstorage[dataInx][2]]])
            next_state_batch = torch.cat((next_state_batch, self.posstorage[dataInx][3]), 0)

        action_batch = torch.as_tensor(action_batch, dtype=torch.float32).to(DEVICE)
        reward_batch = torch.as_tensor(reward_batch, dtype=torch.float32).to(DEVICE)

        return state_batch, action_batch, reward_batch, next_state_batch


    def push_reward(self, reward_flag, reward):
        # 在一局结束后对reward总分存储
        self.rewardflag_dict[reward_flag] = reward

    def get_reward(self, reward_flag):
        return self.rewardflag_dict[reward_flag]


    def Storage_thread(self, state, action, reward_flag, next_state, laststepflag, lastreward):
        # step3 : 抓取动作
        if laststepflag:
            self.push(state, action, lastreward, next_state)
        else:
            self.push(state, action, reward_flag, next_state)


    def length(self):
        print("[*] Length(storage, posstorage, rewardflag_dict):", len(self.storage), len(self.posstorage), len(self.rewardflag_dict))
        return len(self.storage), len(self.posstorage), len(self.rewardflag_dict)


    def save(self):
        print("[-] Replay_buffer saving...")
        if os.path.exists(SUPERHEXAGON_STORE_PATH):
            os.remove(SUPERHEXAGON_STORE_PATH)
        pickle.dump(self.storage, open(SUPERHEXAGON_STORE_PATH, 'wb'))
        if os.path.exists(SUPERHEXAGON_POSSTORE_PATH):
            os.remove(SUPERHEXAGON_POSSTORE_PATH)
        if self.posmode:
            pickle.dump(self.posstorage, open(SUPERHEXAGON_POSSTORE_PATH, 'wb'))
        if os.path.exists(SUPERHEXAGON_REWARDFLAG_PATH):
            os.remove(SUPERHEXAGON_REWARDFLAG_PATH)
        pickle.dump(self.rewardflag_dict, open(SUPERHEXAGON_REWARDFLAG_PATH, 'wb'))
        print("[+] Replay_buffer finish save!")


    def load(self):
        if os.path.exists(SUPERHEXAGON_STORE_PATH):
            self.storage = pickle.load(open(SUPERHEXAGON_STORE_PATH, 'rb'))
            print("[*] REPLAY_BUFFER load finish! len:", len(self.storage))
        if os.path.exists(SUPERHEXAGON_POSSTORE_PATH) and self.posmode:
            self.posstorage = pickle.load(open(SUPERHEXAGON_POSSTORE_PATH, 'rb'))
            print("[*] POSREPLAY_BUFFER load finish! len:", len(self.posstorage))
        if os.path.exists(SUPERHEXAGON_REWARDFLAG_PATH):
            self.rewardflag_dict = pickle.load(open(SUPERHEXAGON_REWARDFLAG_PATH, 'rb'))
            print("[*] REWARDFLAG_BUFFER load finish! len:", len(self.rewardflag_dict))



if __name__ == '__main__':
    replay = AmemoryReplaybuffer()
    replay.load()
    print(replay.storage[0][0].shape, type(replay.storage[0][0]), replay.storage[0][0].device)
    print(replay.rewardflag_dict)
    replay.length()

    state_batch, action_batch, reward_batch, next_state_batch = replay.get(4)
    print(state_batch.shape, action_batch.shape, reward_batch.shape, next_state_batch.shape)