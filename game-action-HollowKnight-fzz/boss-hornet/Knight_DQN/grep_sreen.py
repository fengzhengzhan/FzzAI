# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 12:14:29 2020

@author: pang
"""

import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api
import time
import os

from argconfig import *
from handle_top import handld_top





def grab_screen(region=None):
    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return img

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
        # 大于215判定为一格血量,范围取值，血量面具为高亮
        # print(self_bd_num, end=" ")
        if 50 <= self_bd_num <= 100:
            boss_blood += 1
        # elif boss_blood != 0:
        #     break
    return boss_blood


def test_bloodwindow():
    while True:
        first_screen_grey = grab_screen(blood_window)  # TODO 取差值
        first_screen_grey = cv2.cvtColor(first_screen_grey, cv2.COLOR_RGBA2GRAY)
        print(np.array(first_screen_grey).shape)
        print(self_blood_number(first_screen_grey))
        cv2.imshow("window_main", first_screen_grey)
        cv2.moveWindow("window_main", 700, 540)

        if cv2.waitKey(1) & 0xFF == ord('a'):
            print(cv2.waitKey(1))
            break

    cv2.waitKey()  # 视频结束后，按任意键退出
    cv2.destroyAllWindows()

def test_powerwindow():
    # 大于等于18
    while True:
        first_screen_grey = grab_screen(power_window)  # TODO 取差值
        first_screen_grey = cv2.cvtColor(first_screen_grey, cv2.COLOR_RGBA2GRAY)
        # print(np.array(first_screen_grey).shape)
        print(self_power_number(first_screen_grey, power_window))
        cv2.imshow("window_main", first_screen_grey)
        cv2.moveWindow("window_main", 700, 540)

        if cv2.waitKey(1) & 0xFF == ord('a'):
            print(cv2.waitKey(1))
            break

    cv2.waitKey()  # 视频结束后，按任意键退出
    cv2.destroyAllWindows()

def test_mainwindow():
    while True:
        first_screen_grey = grab_screen(blood_window)  # TODO 取差值
        first_screen_grey = cv2.cvtColor(first_screen_grey, cv2.COLOR_RGBA2GRAY)
        print(self_blood_number(first_screen_grey))
        cv2.imshow("window_main", first_screen_grey)
        cv2.moveWindow("window_main", 700, 540)

        if cv2.waitKey(1) & 0xFF == ord('a'):
            print(cv2.waitKey(1))
            break

    cv2.waitKey()  # 视频结束后，按任意键退出
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # test_bloodwindow()
    # test_powerwindow()
    while True:
        boss_screen_grey = grab_screen(boss_blood_window)
        # print(np.array(boss_screen_grey).shape)
        boss_screen_grey = boss_screen_grey[:,:,2]
        # print(np.array(boss_screen_grey).shape)
        print(boss_blood_number(boss_screen_grey, boss_blood_window))
        cv2.imshow("window_main", boss_screen_grey)
        cv2.moveWindow("window_main", 800, 640)

        if cv2.waitKey(1) & 0xFF == ord('a'):
            print(cv2.waitKey(1))
            break

    cv2.waitKey()  # 视频结束后，按任意键退出
    cv2.destroyAllWindows()
    # left, top, x2, y2
    # main_window = (60, 99, 1219, 610)
    # # blood_window = (222, 95, 530, 120)
    # time.sleep(2)
    # i = 0
    # while True:
    #     i += 1
        # image = grab_screen(main_window)
        # print(image.shape)  # (512, 1160, 4) -> (128, 290) 4倍缩放
        # screen_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # handld_top()
        # first_screen_grey = grab_screen(main_window)  # TODO 取差值
        # first_screen_grey = cv2.cvtColor(first_screen_grey, cv2.COLOR_RGBA2RGB)
        # second_screen_grey = cv2.cvtColor(grab_screen(main_window), cv2.COLOR_BGR2GRAY)
        # blood_window_gray = cv2.cvtColor(grab_screen(blood_window), cv2.COLOR_BGR2GRAY)
        # power_window_gray = cv2.cvtColor(grab_screen(power_window), cv2.COLOR_BGR2GRAY)
        # screen_grey = second_screen_grey - first_screen_grey
        # station = cv2.resize(first_screen_grey, (WIDTH, HEIGHT))
        # second_screen_grey = cv2.cvtColor(grab_screen(main_window), cv2.COLOR_BGR2GRAY)
        # third_screen_grey = cv2.cvtColor(grab_screen(main_window), cv2.COLOR_BGR2GRAY)
        # next_screen_grey = third_screen_grey - second_screen_grey
        # if os.path.exists('.\\label\\knight'+str(i)+'.png'):
        #     pass
        # else:
        #     cv2.imwrite('.\\label\\knight'+str(i)+'.png', first_screen_grey)
        # time.sleep(0.5)
        # cv2.moveWindow("window_main", 700, 540)
        # station = np.array(station).reshape(-1, 1, WIDTH, HEIGHT)
        # self_blood = self_blood_number(blood_window_gray)
        # print(self_blood)
        # self_power = self_power_number(power_window_gray, power_window)

        # print(screen_grey)
        # print(station.shape)




        # print('loop took {} seconds'.format(time.time()))
        # print(self_power_number(screen_grey, power_window))


        # if cv2.waitKey(1) & 0xFF == ord('a'):
        #     print(cv2.waitKey(1))
        #     break
    # cv2.waitKey()  # 视频结束后，按任意键退出
    # cv2.destroyAllWindows()