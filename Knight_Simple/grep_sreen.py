# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 12:14:29 2020

@author: pang
"""

import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api
import time

from argconfig import *

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


if __name__ == '__main__':
    # left, top, x2, y2
    # main_window = (60, 99, 1219, 610)
    # blood_window = (222, 95, 530, 120)
    while True:
        # 测试代码 查看窗口位置
        second_screen_grey = cv2.cvtColor(grab_screen(main_window), cv2.COLOR_BGR2GRAY)
        cv2.imshow("window_main", second_screen_grey)
        cv2.moveWindow("window_main", 700, 540)
        if cv2.waitKey(1) & 0xFF == ord('a'):
            print(cv2.waitKey(1))
            break
    cv2.waitKey()  # 视频结束后，按任意键退出
    cv2.destroyAllWindows()