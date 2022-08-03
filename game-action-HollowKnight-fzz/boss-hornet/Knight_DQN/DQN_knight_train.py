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


def pause_game(paused):
    if paused:
        print("[-] paused")
        while True:
            keys = key_check()
            if 'T' in keys:
                if paused:
                    paused = False
                    print("[+] start game")
                    esc_quit()
                    time.sleep(1)
                    break
                else:
                    paused = True
                    esc_quit()
                    time.sleep(1)# jw
    keys = key_check()
    if 'T' in keys:
        if paused:
            paused = False
            print("[+] start game")
            esc_quit()
            time.sleep(1)
        else:
            paused = True
            print("[-] pause game")
            esc_quit()
            time.sleep(1)

    return paused

def self_blood_number(self_gray):
    self_blood = 0
    range_set = False
    # print(self_gray[0])
    # print(self_gray, self_gray.shape)
    for self_bd_num in self_gray[0]:
        # 大于215判定为一格血量,范围取值，血量面具为高亮
        # print(self_bd_num, end=" ")
        if self_bd_num > 215 and range_set:
            self_blood += 1
            range_set = False
        elif self_bd_num < 55:
            range_set = True
    # print(self_blood)
    return self_blood

def self_power_number(self_gray, power_window):
    self_power = 0
    # range_set = True
    # print(self_gray[0])
    # print(self_gray, self_gray.shape)
    # print(self_gray)
    for i in range(0, power_window[3] - power_window[1]):
        self_power_num = self_gray[i][0]
        # print(self_power_num, end=" ")
        # 大于52(60)就给予正反馈
        # print(self_bd_num)
        if self_power_num > 90:
            self_power += 1
    # print(self_blood)
    return self_power

def boss_blood_number(self_rimg, boss_blood_window):
    boss_blood = 0
    for self_bd_num in self_rimg[0]:
        # print(self_bd_num, end=" ")
        if 53 <= self_bd_num <=100:
            boss_blood += 1
        # elif boss_blood != 0:
        #     break
    return boss_blood

ch_attack_action = ['上击', '下击', '攻击', '双击', '放波', '上吼', '下砸', ]
ch_move_action = ['左走', '右走', '冲刺', '短跳', '向左', '向右', '长跳', ]

def take_attack_action(action):
    # 所有的攻击
    if action == I_ATTACK:
        i_attack()
    elif action == K_ATTACK:
        k_attack()
    elif action == ATTACK_NUM:
        r_short_attack()
    elif action == ATTACK_DOUBLE_NUM:
        r_doubleshort_attack()
    elif action == Q_SHORT_USE:
        q_short_use()
    elif action == IQ_BOOM:
        iq_boom()
    elif action == KQ_BOOM:
        kq_doom()
    # elif action == Q_LONG_USE:
    #     q_long_use()

def take_move_action(action, last_action):
    # 所有移动
    if last_action == TO_LEFT:
        ReleaseKey(J)
        time.sleep(0.02)
    elif last_action == TO_RIGHT:
        ReleaseKey(L)
        time.sleep(0.02)
    elif last_action == SPACE_STAY:
        ReleaseKey(SPACE)
        time.sleep(0.02)

    if action == J_LEFT:
        j_left()
    elif action == L_RIGHT:
        l_right()
    elif action == E_RUN:
        e_run()
    elif action == 3:
        space_short_up()
    elif action == TO_LEFT:
        to_left()
    elif action == TO_RIGHT:
        to_right()
    elif action == SPACE_STAY:
        space_stay()




if __name__ == '__main__':

    if not os.path.exists("model"):
        os.mkdir("model")

    # use_tree = SumTree(REPLAY_SIZE)
    # judge = JUDGE(use_tree)
    judge = JUDGE()

    policy_net = DQN(ACTIONATTACK_SIZE, ACTIONMOVE_SIZE).to(DEVICE)
    # policymove_net = DQNMove(WIDTH, HEIGHT, ACTIONMOVE_SIZE).to(DEVICE)

    for m in policy_net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")  # kaiming初始化

    if os.path.exists(DQN_MODEL_PATH) and os.path.exists(DQN_MODELMOVE_PATH):
        policy_net.load_state_dict(torch.load(DQN_MODEL_PATH))
        # policymove_net.load_state_dict(torch.load(DQN_MODELMOVE_PATH))
        print("[*] policy_net policymove_net load finish!")
    # target_net = DQN(WIDTH, HEIGHT, ACTIONATTACK_SIZE).to(DEVICE)
    # targetmove_net = DQNMove(WIDTH, HEIGHT, ACTIONMOVE_SIZE).to(DEVICE)
    # target_net.load_state_dict(policy_net.state_dict())
    # targetmove_net.load_state_dict(policymove_net.state_dict())
    # target_net.eval()
    # targetmove_net.eval()
    # print("[*] target_net", target_net.eval())


    if os.path.exists(DQN_STORE_PATH):
        judge.replay_buffer = pickle.load(open(DQN_STORE_PATH, 'rb'))
        print("[*] REPLAY_BUFFER load finish! len:", len(judge.replay_buffer))
    # if len(judge.replay_buffer) != REPLAY_SIZE:
    #     use_tree.write = len(judge.replay_buffer)


    plt_step_list = []
    plt_step = 0
    plt_reward = []
    plt.ion()
    plt.figure(1, figsize=(10, 1))

    plt.plot(plt_step_list, plt_reward, color="orange")
    plt.pause(3)

    # DQN init
    paused = True
    paused = pause_game(paused)
    # emergence_break = 0  # 用于防止错误训练数据扰乱神经网络
    # target_step = 0
    step = 0
    training_num = TRAINING_NUM

    TRAIN_FLAG = True
    boss_blood_save = 360
    boss_blood_save_flag = 360

    handld_top()
    init_start()

    for episode in range(1, EPISODES):
        done = 0
        total_reward = 0
        avg_step = 1



        blood_window_gray_first = cv2.cvtColor(grab_screen(blood_window), cv2.COLOR_RGBA2GRAY)
        # power_window_gray = cv2.cvtColor(grab_screen(power_window), cv2.COLOR_RGBA2GRAY)
        judge.all_blood = self_blood_number(blood_window_gray_first)
        judge.boss_blood = BOSS_ALL_BLOOD
        # judge.all_power = self_power_number(power_window_gray, power_window)
        choose_time = time.time()

        judge.all_loss = torch.tensor(0.)
        training_num += 1

        pass_attack_action = -1
        pass_move_action = -1

        boss_blood_flag = True

        last_action = -1

        # last_time = time.time()
        init_time = time.time()

        while True:
            # cv2.imshow("window_main", next_station)
            # cv2.moveWindow("window_main", 1100, 540)
            # if cv2.waitKey(1) & 0xFF == ord('a'):
            #     pass
            for tmp in range(ONE_ATTACK):
                # step1 : 首次抓取
                first_screen_grey = grab_screen(main_window)
                first_screen_grey = cv2.cvtColor(first_screen_grey, cv2.COLOR_RGBA2RGB)
                first_screen_grey = np.transpose(first_screen_grey, (2, 0, 1))  # Tensor通道排列顺序是：[batch, channel, height, width]
                station = first_screen_grey[np.newaxis,:]
                # station = torch.as_tensor(station, dtype=torch.float32).to(DEVICE)

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
                # print("[*] station:", station)
                move_action, attack_action, action_choose_name = judge.choose_action(policy_net, station, training_num)

                handld_top()
                take_move_action(move_action, last_action)
                take_attack_action(attack_action)
                last_action = move_action


                # step3 : 抓取动作
                third_screen_grey = grab_screen(main_window)
                third_screen_grey = cv2.cvtColor(third_screen_grey, cv2.COLOR_RGBA2RGB)
                third_screen_grey = np.transpose(third_screen_grey, (2, 0, 1))
                next_station = third_screen_grey[np.newaxis, :]
                # next_station = torch.as_tensor(next_station, dtype=torch.float32).to(DEVICE)

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
                reward, done, boss_blood_display = judge.action_judge(next_self_blood, next_boss_blood, attack_action, move_action, pass_attack_action, pass_move_action, self_power)

                judge.store_data(station, attack_action, move_action, reward, next_station)

                pass_attack_action = attack_action
                pass_move_action = move_action

                total_reward += reward
                avg_step += 1

                print('once {} {} {} boss{} reward {} {} {}.'.format(
                    ch_move_action[move_action], ch_attack_action[attack_action], action_choose_name, boss_blood_display,
                    reward, total_reward, time.time()-init_time))
                init_time = time.time()
                # last_time = time.time()
                # judge.train_network(policy_net, BATCH_SIZE, True, done, num_step)


                if done == 1:
                    boss_blood_save = next_boss_blood
                    blood_window_gray_done = cv2.cvtColor(grab_screen(blood_window), cv2.COLOR_RGBA2GRAY)
                    self_blood_done = self_blood_number(blood_window_gray_done)
                    if self_blood_done == 0:
                        PressKey(J)
                        PressKey(L)
                        PressKey(SPACE)
                        judge.choose_action_time = time.time() - choose_time
                        break

                # print("[S] station:", station[0], station[0].shape) # t
                # print("[S] action:", action)
                # print("[S] reward:", reward)
                # print("[S] next_station:", next_station[0], next_station[0].shape)

                # get action reward
                # if emergence_break == 100:
                #     print("[-] emergence_break")
                #     # save_model(policy_net, DQN_MODEL_PATH)
                #     paused = True

            # target_step += 1

            if len(judge.replay_buffer) > STORE_SIZE:
                num_step += 1
                judge.train_network(policy_net, BATCH_SIZE, True, done, num_step)
                # judge.train_network(policy_net, target_net, policymove_net, targetmove_net, BATCH_SIZE, True, done, num_step)
            # if target_step % UPDATE_STEP == 0:
            #     target_net.load_state_dict(policy_net.state_dict())
            #     targetmove_net.load_state_dict(policymove_net.state_dict())

            # station = next_station
            # self_blood = next_self_blood
            # total_reward += reward
            # avg_step += 1

            paused = pause_game(paused)

            # print('once {} {} reward {}, second{}.'.format(ch_action[action], action_choose_name, total_reward, time.time() - last_time))
            # last_time = time.time()

            if done == 1:
                boss_blood_save = next_boss_blood
                blood_window_gray_done = cv2.cvtColor(grab_screen(blood_window), cv2.COLOR_RGBA2GRAY)
                self_blood_done = self_blood_number(blood_window_gray_done)
                if self_blood_done == 0:
                    PressKey(J)
                    PressKey(L)
                    PressKey(SPACE)
                    judge.choose_action_time = time.time() - choose_time
                    break

        # target_net.load_state_dict(policy_net.state_dict())
        # targetmove_net.load_state_dict(policymove_net.state_dict())
        judge.train_network(policy_net, ALL_BATCH_SIZE, True, done, num_step)
        if episode % 30 == 0:
            print("[-] saving...")
            torch.save(policy_net.state_dict(), DQN_MODEL_PATH)
            # torch.save(policymove_net.state_dict(), DQN_MODELMOVE_PATH)
            if os.path.exists(DQN_STORE_PATH):
                os.remove(DQN_STORE_PATH)
            pickle.dump(judge.replay_buffer, open(DQN_STORE_PATH, 'wb'))
            # use_tree.stroe_tree()
            print("[+] Finish save!")
        if boss_blood_save_flag - boss_blood_save >= 20:
            boss_blood_save_flag = boss_blood_save
            print("[-] "+str(boss_blood_save)+" saving...")
            torch.save(policy_net.state_dict(), "E:\\1_mycode\\Knight_DQN\\model\\dqn_model"+str(boss_blood_save)+str(time.strftime("%Y%m%d%H%M%S", time.localtime()))+".pt")
            # torch.save(policymove_net.state_dict(), "E:\\1_mycode\\Knight_DQN\\model\\dqn_movemodel"+str(boss_blood_save)+str(time.strftime("%Y%m%d%H%M%S", time.localtime()))+".pt")
            # use_tree.stroe_tree()
            print("[+] "+str(boss_blood_save)+" Finish save!")
        plt_step_list.append(plt_step)
        plt_step += 1
        plt_reward.append(judge.all_loss.detach().numpy() / avg_step)
        plt.plot(plt_step_list, plt_reward, color="orange")
        plt.pause(0.1)
        if len(plt_step_list) >= 800:
            plt_step_list = []
            plt_reward = []
        print("[*] Epoch: ", episode, "Store: ", judge.position if len(judge.replay_buffer) == REPLAY_SIZE else len(judge.replay_buffer),
              "Loss: ", judge.all_loss.detach().numpy() / avg_step, "Time: ", judge.choose_action_time)

        if judge.position % REPLAY_SIZE <= 140 and judge.position != 0:
        # if judge.position % 4000 == 0:
            step_loss = torch.tensor(0.)
            for i in range(200):
                step += 1
                judge.all_loss = torch.tensor(0.)
                judge.train_network(policy_net, ALL_BATCH_SIZE, False, 0, num_step)
                plt_step_list.append(plt_step)
                plt_step += 1
                plt_reward.append(judge.all_loss.detach().numpy())
                plt.plot(plt_step_list, plt_reward, color="orange")
                plt.pause(0.1)
                if len(plt_step_list) >= 800:
                    plt_step_list = []
                    plt_reward = []
                print("[*] Replaying: ", step, "Loss: ", judge.all_loss.detach().numpy())
                step_loss += judge.all_loss

                # if step % UPDATE_STEP == 0:
                #     target_net.load_state_dict(policy_net.state_dict())
                #     targetmove_net.load_state_dict(policymove_net.state_dict())

                if step % 20 == 0:
                    print("[-] saving...")
                    torch.save(policy_net.state_dict(), DQN_MODEL_PATH)
                    # torch.save(policymove_net.state_dict(), DQN_MODELMOVE_PATH)
                    print("[+] Finish save!")
                    judge.train_network(policy_net, ALL_BATCH_SIZE, False, 1, num_step)
                    if (step_loss / 20.0) <= 500:
                        break
                    else:
                        step_loss = torch.tensor(0.)


        time.sleep(2)
        handld_top()
        init_start()




