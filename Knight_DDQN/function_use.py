import os
import time
import cv2
import numpy as np
import random
import os
import pickle
from handle_top import handld_top

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
    ReleaseKey(J)
    ReleaseKey(L)
    ReleaseKey(SPACE)
    time.sleep(0.02)

    if action == J_LEFT:
        j_left()
    elif action == L_RIGHT:
        l_right()
    elif action == E_RUN:
        e_run()
    elif action == SPACE_SHORT:
        space_short_up()
    elif action == TO_LEFT:
        to_left()
    elif action == TO_RIGHT:
        to_right()
    elif action == SPACE_STAY:
        space_stay()