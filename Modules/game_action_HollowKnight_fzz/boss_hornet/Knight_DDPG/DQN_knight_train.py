import os
import time
import cv2
import numpy as np
import random
import os
import pickle
from handle_top import handld_top

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt

from keyboard_use import *
from DQN_pytorch_gpu import *
from getkeys import key_check
from keyboard_use import *
from grep_sreen import grab_screen
from argconfig import *
from function_use import *


import threading


# DDPG训练模型
agent = DDPG()
agent.load()
# 经验池
replay = Replay_buffer()
replay.load()



# while True:
#     agent.train_network(replay, ALL_BATCH_SIZE, True, 0, num_step)

def Storage_thread(station, move_action, attack_action, reward):
    global replay
    # step3 : 抓取动作
    third_screen_grey = grab_screen(main_window)
    third_screen_grey = cv2.cvtColor(third_screen_grey, cv2.COLOR_RGBA2RGB)
    third_screen_grey = np.transpose(third_screen_grey, (2, 0, 1))
    next_station = third_screen_grey[np.newaxis, :]

    replay.push(station, move_action, attack_action, reward, next_station)

def select_action(x):
    # Vector
    # 处理一维数组
    x_max = np.max(x)
    x -= x_max
    numerator = np.exp(x)
    denominator = 1.0 / np.sum(numerator)
    x = numerator.dot(denominator)

    x = torch.as_tensor(x)  # 生成分布
    x_m = Categorical(x)  # 从分布中采样
    x_a = x_m.sample()

    return x_a.item()




plt_step_list = []
plt_step = 0
plt_loss = []
plt.ion()
plt.figure(1, figsize=(10, 1))
plt.plot(plt_step_list, plt_loss, color="orange")
plt.pause(3)

# DQN init
paused = True
paused = pause_game(paused)


TRAIN_FLAG = True
boss_blood_save = 360
boss_blood_save_flag = 360

# if replay.ptr <= 140 and len(replay.storage) >= STORE_SIZE:
#     step_loss = torch.tensor(0.)
#     print("[*] Replay len:", len(replay.storage))
#     for i in range(200):
#         step += 1
#         agent.all_loss = torch.tensor(0.)
#         agent.train_network(replay, ALL_BATCH_SIZE, False, 0, num_step)
#         plt_step_list.append(plt_step)
#         plt_step += 1
#         plt_loss.append(agent.all_loss.detach().numpy())
#         plt.plot(plt_step_list, plt_loss, color="orange")
#         plt.pause(0.1)
#         if len(plt_step_list) >= 800:
#             plt_step_list = []
#             plt_loss = []
#         print("[*] Replaying: ", step, "Loss: ", agent.all_loss.detach().numpy())
#         step_loss += agent.all_loss
#
#         if step % 40 == 0:
#             agent.train_network(replay, BATCH_SIZE, True, 0, num_step)
#             agent.save()
#             if (step_loss / 40.0) <= 30:
#                 break
#             else:
#                 step_loss = torch.tensor(0.)

handld_top()
init_start()

for episode in range(1, EPISODES):
    step = 0
    done = 0
    done_is = True
    total_reward = 0
    avg_step = 1

    blood_window_gray_first = cv2.cvtColor(grab_screen(blood_window), cv2.COLOR_RGBA2GRAY)
    # power_window_gray = cv2.cvtColor(grab_screen(power_window), cv2.COLOR_RGBA2GRAY)
    agent.all_blood = self_blood_number(blood_window_gray_first)
    agent.boss_blood = BOSS_ALL_BLOOD
    # agent.all_power = self_power_number(power_window_gray, power_window)

    agent.all_loss = torch.tensor(0.)

    # pass_move_action = -1
    # pass_attack_action = -1

    boss_blood_flag = True

    last_move_num = -1

    # init_time = time.time()


    while True:

        # step1 : 首次抓取
        first_screen_grey = grab_screen(main_window)
        first_screen_grey = cv2.cvtColor(first_screen_grey, cv2.COLOR_RGBA2RGB)
        first_screen_grey = np.transpose(first_screen_grey, (2, 0, 1))  # Tensor通道排列顺序是：[batch, channel, height, width]
        station = first_screen_grey[np.newaxis,:]

        blood_window_gray = cv2.cvtColor(grab_screen(blood_window), cv2.COLOR_RGBA2GRAY)
        boss_screen_grey = grab_screen(boss_blood_window)
        boss_screen_grey = boss_screen_grey[:, :, 2]
        power_window_gray = cv2.cvtColor(grab_screen(power_window), cv2.COLOR_RGBA2GRAY)
        self_blood = self_blood_number(blood_window_gray)
        boss_blood = boss_blood_number(boss_screen_grey, boss_blood_window)
        if boss_blood <= 500 and boss_blood_flag:
            boss_blood = BOSS_ALL_BLOOD
        else:
            boss_blood_flag = False
        self_power = self_power_number(power_window_gray, power_window)

        # step2 : 执行动作
        move_batch, attack_batch = agent.choose_action(station)

        move_num = select_action(move_batch)
        attack_num = select_action(attack_batch)

        handld_top()
        take_move_action(move_num, last_move_num)
        take_attack_action(attack_num)



        # step3 : 抓取动作
        next_blood_window_gray = cv2.cvtColor(grab_screen(blood_window), cv2.COLOR_RGBA2GRAY)
        # next_power_window_gray = cv2.cvtColor(grab_screen(power_window), cv2.COLOR_RGBA2GRAY)
        next_boss_screen_grey = grab_screen(boss_blood_window)
        next_boss_screen_grey = next_boss_screen_grey[:, :, 2]

        next_self_blood = self_blood_number(next_blood_window_gray)
        next_boss_blood = boss_blood_number(next_boss_screen_grey, boss_blood_window)
        if next_boss_blood <= 500 and boss_blood_flag:
            next_boss_blood = BOSS_ALL_BLOOD
        else:
            boss_blood_flag = False
        # next_self_power = self_power_number(next_power_window_gray, power_window) #

        # print("[*] self_blood next_self_blood", ALL_BLOOD, self_blood, next_self_blood)
        reward, done, boss_blood_display = agent.action_judge(next_self_blood, next_boss_blood, move_num, attack_num, self_power, last_move_num, done_is)

        last_move_num = move_num

        # 多线程存储
        threading.Thread(target=Storage_thread, args=(station, move_batch, attack_batch, reward)).start()


        # print('once {} {} boss{} reward {} {} {}.'.format(
        #     ch_move_action[move_num], ch_attack_action[attack_num],
        #     boss_blood_display,
        #     reward, total_reward, time.time() - init_time))
        # init_time = time.time()
        # agent.train_network(replay, ALL_BATCH_SIZE, True, 0, num_step)
        print('once {} {} boss{} reward {} {}.'.format(ch_move_action[move_num], ch_attack_action[attack_num], boss_blood_display, reward, total_reward))


        total_reward += reward
        avg_step += 1


        # paused = pause_game(paused)

        if done == 1:
            boss_blood_save = next_boss_blood
            self_blood_done_flag = True
            done = 0
            done_is = False
            for i in range(JUDGE_DONE):
                blood_window_gray_done = cv2.cvtColor(grab_screen(blood_window), cv2.COLOR_RGBA2GRAY)
                self_blood_done = self_blood_number(blood_window_gray_done)
                if self_blood_done != 0:
                    self_blood_done_flag = False
                    break
            if self_blood_done_flag:
                PressKey(J)
                PressKey(L)
                PressKey(SPACE)
                break

    if episode % 30 == 0:
        agent.save()
        replay.save()

    if boss_blood_save_flag - boss_blood_save >= 20:
        boss_blood_save_flag = boss_blood_save
        agent.save(str(boss_blood_save))

    plt_step_list.append(plt_step)
    plt_step += 1
    plt_loss.append(agent.all_loss.detach().numpy() / avg_step)
    plt.plot(plt_step_list, plt_loss, color="orange")
    plt.pause(0.1)
    if len(plt_step_list) >= 800:
        plt_step_list = []
        plt_loss = []
    print("[*] Epoch: ", episode, "Store: ", replay.ptr, "Loss: ", agent.all_loss.detach().numpy() / avg_step)

    if replay.ptr <= 140 and replay.ptr != 0 and len(replay.storage) >= STORE_SIZE:
        step_loss = torch.tensor(0.)
        replay.save()
        print("[*] Replay len:", len(replay.storage))
        for i in range(200):
            step += 1
            agent.all_loss = torch.tensor(0.)
            agent.train_network(replay, ALL_BATCH_SIZE, False, 0, num_step)
            plt_step_list.append(plt_step)
            plt_step += 1
            plt_loss.append(agent.all_loss.detach().numpy())
            plt.plot(plt_step_list, plt_loss, color="orange")
            plt.pause(0.1)
            if len(plt_step_list) >= 800:
                plt_step_list = []
                plt_loss = []
            print("[*] Replaying: ", step, "Loss: ", agent.all_loss.detach().numpy())
            step_loss += agent.all_loss

            if step % 40 == 0:
                agent.train_network(replay, BATCH_SIZE, True, 0, num_step)
                agent.save()
                if (step_loss / 40.0) <= 30:
                    break
                else:
                    step_loss = torch.tensor(0.)


    time.sleep(2)
    handld_top()
    init_start()

