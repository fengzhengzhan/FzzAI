# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 09:45:04 2020

@author: pang
"""

import numpy as np
from PIL import ImageGrab
import cv2
import time
import grep_sreen
import os

def self_blood_number(self_gray):
    self_blood = 0
    range_set = True
    # print(self_gray[0])
    # print(self_gray, self_gray.shape)
    for self_bd_num in self_gray[0]:
        # 大于215判定为一格血量,范围取值，血量面具为高亮
        # print(self_bd_num)
        if self_bd_num > 216 and range_set:
            self_blood += 1
            range_set = False
        elif self_bd_num < 100:
            range_set = True
    print(self_blood)
    return self_blood


main_window = (60, 99, 1219, 610)
blood_window = (222, 95, 530, 120)



last_time = time.time()
while(True):

    #printscreen = np.array(ImageGrab.grab(bbox=(window_size)))
    #printscreen_numpy = np.array(printscreen_pil.getdata(),dtype='uint8')\
    #.reshape((printscreen_pil.size[1],printscreen_pil.size[0],3))
    #pil格式耗时太长
    
    screen_gray = cv2.cvtColor(grep_sreen.grab_screen(main_window), cv2.COLOR_BGR2GRAY)#灰度图像收集
    # screen_reshape = cv2.resize(screen_gray,(96,86))
    self_blood = self_blood_number(screen_gray)
    print(screen_gray.shape)
    screen_gray = cv2.resize(screen_gray, (290, 128))
    print(screen_gray.shape)
    
    cv2.imshow('window_blood', screen_gray)
    cv2.moveWindow("window_blood", 1200, 100)
    #cv2.imshow('window3',printscreen)
    #cv2.imshow('window2',screen_reshape)
    
    #测试时间用
    print('loop took {} seconds'.format(time.time()-last_time))
    last_time = time.time()
    
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cv2.waitKey()# 视频结束后，按任意键退出
cv2.destroyAllWindows()
