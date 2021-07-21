from multiprocessing import Process, Manager
import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api
from config import *
import time
import torch


# 自定义进程类，以便训练动作同时抓取屏幕观察
class GrabScreenProcess(Process):
    def __init__(self, screen_index, screen_list):
        super(GrabScreenProcess, self).__init__()
        self.screen_index = screen_index
        self.screen_list = screen_list

    # 屏幕截取函数
    def grab_screen(self, region=None):
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

    # 将图像使用opencv处理，缩小，去色
    def image_conversion(self, img):
        res = cv2.resize(img, RESIZE_WINDOW, interpolation=cv2.INTER_AREA)
        grayimg = cv2.cvtColor(res, cv2.COLOR_RGBA2RGB)
        # grayimg = cv2.convertScaleAbs(grayimg, alpha=2.5, beta=0)  # 增加对比度
        return grayimg

    # run()是Process类专门留出来用于重写的接口函数
    def run(self):
        while True:
            img = self.grab_screen(region=SUPERHEXAGON_WINDOW)
            # print(img.shape)
            # cv2.imshow("window_main", img)
            # cv2.moveWindow("window_main", 800, 340)
            # if cv2.waitKey(1) & 0xFF == ord('a'):
            #     print(cv2.waitKey(1))
            #     break
            img = self.image_conversion(img)
            img = np.transpose(img, (2, 0, 1))
            torchimg = img[np.newaxis, :]  #(C,H,W) 3D卷积核(N,C,D,H,W)
            torchimg = torch.as_tensor(torchimg, dtype=torch.float32).to(CPUDEVICE)
            # print(torchimg.shape)
            # print(time.time(), self.screen_index.value)  # DEBUG

            # 滚动数组
            # print("debug:", (self.screen_index.value+1) % MANAGER_LIST_LENGTH)
            self.screen_list[(self.screen_index.value+1) % MANAGER_LIST_LENGTH] = torchimg
            self.screen_index.value += 1
            # time.sleep(SLEEP_SCREEN)

class AeyeGrabscreen(object):
    def __init__(self):
        super(AeyeGrabscreen, self).__init__()
        # 进程间共享变量
        manager = Manager()
        self.screen_index = manager.Value('i', MANAGER_LIST_LENGTH)  # 为防止数组越界
        self.screen_list = manager.list([0 for i in range(MANAGER_LIST_LENGTH)])

        grabscreen_process = GrabScreenProcess(screen_index=self.screen_index, screen_list=self.screen_list)
        grabscreen_process.start()  # 开启进程
        time.sleep(SLEEP_INIT_GRAB)

    def getstate(self):
        x = int(self.screen_index.value)
        # state = torch.stack([
        #     self.screen_list[(x - 3) % MANAGER_LIST_LENGTH],
        #     self.screen_list[(x - 2) % MANAGER_LIST_LENGTH],
        #     self.screen_list[(x - 1) % MANAGER_LIST_LENGTH],
        #     self.screen_list[x % MANAGER_LIST_LENGTH]
        # ], dim=1)
        state = self.screen_list[x % MANAGER_LIST_LENGTH]
        state = torch.as_tensor(state, dtype=torch.float32).to(DEVICE)
        # print(state.shape, x)
        return state



if __name__ == '__main__':
    aeyegrabscreen = AeyeGrabscreen()
    while True:
        state = aeyegrabscreen.getstate()
        # state = state - state
        # time.sleep(0.2)
        # next_state = aeyegrabscreen.getstate()
        print(state, state.shape)

    # while True:
    #     print(screen_index)
    #     cv2.imshow("window_main", screen_list[0])
    #     cv2.moveWindow("window_main", 800, 340)
    #     if cv2.waitKey(1) & 0xFF == ord('a'):
    #         print(cv2.waitKey(1))
    #         break
