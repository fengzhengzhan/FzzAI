from multiprocessing import Process, Manager
import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api
from config import *
import time
import torch
import math


# 自定义进程类，以便训练动作同时抓取屏幕观察
class GrabScreenProcess(Process):
    def __init__(self, screen_index, screen_list, area_list):
        super(GrabScreenProcess, self).__init__()
        self.screen_index = screen_index
        self.screen_list = screen_list
        self.area_list = area_list

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
    def self_image_conversion(self, img):
        img = cv2.resize(img, RESIZE_WINDOW, interpolation=cv2.INTER_AREA)
        grayimg = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        # grayimg = cv2.convertScaleAbs(grayimg, alpha=2.5, beta=0)  # 增加对比度
        _, grayimg = cv2.threshold(grayimg, 128, GRAY_IMAGE_COLOR, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # print(grayimg.shape)  (214, 284) (h, w)

        # 轮廓检测，获取更多信息
        tri_center = []
        sixpoint_center = []
        sixpoint_flag = True
        barrierfourpoint_list = []
        barriermorepoint_list = []
        danger_state = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

        # 检测轮廓， 找出凸包, 得到所有的点的信息
        contours, hierarchy = cv2.findContours(grayimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print("[*]", np.array(contours).shape, np.array(hierarchy).shape)
        for c in contours:
            # print(np.array(c).shape, len(c))
            # grayimg = cv2.drawContours(grayimg, c, -1, (0, 255, 0), 3)  # 矩形标记
            x, y, w, h = cv2.boundingRect(c)
            # cv2.rectangle(grayimg, (x, y), (x + w, y + h), (255, 255, 0), 1)  # 画矩形框
            # 找出凸包
            hull = cv2.convexHull(c)
            epsilon = 0.02 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            approx = approx.reshape((len(approx), 2))

            if 20 <= w * h <= 50:  # 方块位置
                tri_center = np.array(c).reshape(-1, 2).mean(axis=0)
                # print(tri_center)
                continue

            elif 1400 <= w * h <= 3000 and len(approx) == 6 and sixpoint_flag:  # 定位六边形（六个点坐标）
                sixpoint_flag = False
                tmp = approx[0]
                for idx, point in enumerate(approx[1:]):
                    sixpoint_center.append([(point[0]+tmp[0])/2, (point[1]+tmp[1])/2])
                    tmp = point
                    if idx == len(approx[1:])-1:
                        sixpoint_center.append([(point[0] + approx[0][0]) / 2, (point[1] + approx[0][1]) / 2])
                continue
                # for i, p in enumerate(sixpoint_center):
                #     if i == 0:
                #         print(p)
                #         grayimg[int(p[1])][int(p[0])] = 255.0

            elif len(approx) == 4:  # 四点规则障碍物对应的障碍物位置, 计算中心点
                barrierfourpoint_dict = {}
                tmp = approx[0]
                for idx, point in enumerate(approx[1:]):
                    barrierfourpoint_dict[math.sqrt( (point[0]-tmp[0])**2 + (point[1]-tmp[1])**2 )] = [(point[0] + tmp[0]) / 2, (point[1] + tmp[1]) / 2]
                    tmp = point
                    if idx == len(approx[1:])-1:
                        barrierfourpoint_dict[math.sqrt((point[0] - approx[0][0]) ** 2 + (point[1] - approx[0][1]) ** 2)] = [(point[0] + approx[0][0]) / 2, (point[1] + approx[0][1]) / 2]
                barrierfourpoint_list.append(barrierfourpoint_dict)
                # cv2.rectangle(grayimg, (x, y), (x + w, y + h), (255, 255, 0), 1)  # 画矩形框
                # print(barrierfourpoint_dict)
                # time.sleep(0.1)
                # print(approx)

            elif len(approx) > 4:  # 对于多边形障碍，只需要求出相邻边障碍，不需要封闭
                barriermorepoint_dict = {}
                tmp = approx[0]
                try:
                    for idx, point in enumerate(approx[1:]):
                        # 只有当中心点周围为0.0时才能加入字典
                        if grayimg[int((point[1] + tmp[1]) / 2)][int((point[0] + tmp[0]) / 2)] < 0.5 \
                                and grayimg[int((point[1] + tmp[1]) / 2) - 1][int((point[0] + tmp[0]) / 2) ] < 0.5 \
                                and grayimg[int((point[1] + tmp[1]) / 2) - 2][int((point[0] + tmp[0]) / 2) ] < 0.5 \
                                and grayimg[int((point[1] + tmp[1]) / 2) + 1][int((point[0] + tmp[0]) / 2) ] < 0.5 \
                                and grayimg[int((point[1] + tmp[1]) / 2) + 2][int((point[0] + tmp[0]) / 2) ] < 0.5 \
                                and grayimg[int((point[1] + tmp[1]) / 2) ][int((point[0] + tmp[0]) / 2) - 1] < 0.5 \
                                and grayimg[int((point[1] + tmp[1]) / 2) ][int((point[0] + tmp[0]) / 2) - 2] < 0.5 \
                                and grayimg[int((point[1] + tmp[1]) / 2) ][int((point[0] + tmp[0]) / 2) + 1] < 0.5 \
                                and grayimg[int((point[1] + tmp[1]) / 2) ][int((point[0] + tmp[0]) / 2) + 2] < 0.5:
                            pass
                        else:
                            barriermorepoint_dict[math.sqrt((point[0] - tmp[0]) ** 2 + (point[1] - tmp[1]) ** 2)] = [(point[0] + tmp[0]) / 2, (point[1] + tmp[1]) / 2]
                            # grayimg[int((point[1] + tmp[1]) / 2)][int((point[0] + tmp[0]) / 2)] = 255.0
                        tmp = point
                        # print(int(tmp[0]), int(tmp[1]), int(point[0]), int(point[1]), int((point[0] + tmp[0]) / 2), int((point[1] + tmp[1]) / 2))
                        # if idx == len(approx[1:])-1:
                        #     barriermorepoint_dict[math.sqrt((point[0] - approx[0][0]) ** 2 + (point[1] - approx[0][1]) ** 2)] = [(point[0] + approx[0][0]) / 2, (point[1] + approx[0][1]) / 2]
                except Exception as e:
                    pass
                barriermorepoint_list.append(barriermorepoint_dict)
                # print(barriermorepoint_dict)
                # cv2.rectangle(grayimg, (x, y), (x + w, y + h), (255, 255, 0), 1)  # 画矩形框


        # 对六个方位进行判断[左三区是否有障碍(有1.0 无0.2), ..., ]
        if len(tri_center) != 0 and len(sixpoint_center) != 0:
            # 判断中心点所在区域
            area_idx = -1
            tmp_length = 40000
            for idx, point in enumerate(sixpoint_center):
                length = math.sqrt( (point[0]-tri_center[0])**2 + (point[1]-tri_center[1])**2 )
                if length < tmp_length:
                    area_idx = idx
                    tmp_length = length
            # grayimg[int(tri_center[1])][int(tri_center[0])] = 255.0
            # grayimg[int(sixpoint_center[area_idx][1])][int(sixpoint_center[area_idx][0])] = 255.0

            # 各区域的index
            # 左三 -> 右二  (area_idx+3)%6 (area_idx+4)%6 (area_idx+5)%6 area_idx (area_idx+1)%6 (area_idx+2)%6
            # 判断区域位置是否有障碍
            # 根据距离，使用单个长度
            area_idx_list = [(area_idx+3)%6, (area_idx+4)%6, (area_idx+5)%6, area_idx, (area_idx+1)%6, (area_idx+2)%6]

            # 四边的根据中心点求所在区域
            for idx, dic in enumerate(barrierfourpoint_list):
                # print(idx, dic)
                tmp_four_x = 0
                tmp_four_y = 0
                for k, v in dic.items():
                    tmp_four_x += v[0]
                    tmp_four_y += v[1]
                tmp_four_x = tmp_four_x / 4
                tmp_four_y = tmp_four_y / 4
                # grayimg[int(tmp_four_y)][int(tmp_four_x)] = 255.0
                # 寻找区域
                fouraid = 0
                fourdistance = 40000
                for i, aid in enumerate(area_idx_list):
                    tmp_distance = math.sqrt( (tmp_four_x - sixpoint_center[aid][0])**2 + (tmp_four_y - sixpoint_center[aid][1])**2 )
                    if tmp_distance < fourdistance:
                        fouraid = aid
                        fourdistance = tmp_distance
                if FOURBARRIER_DISTANCE[0] <= fourdistance <= FOURBARRIER_DISTANCE[1]:
                    danger_state[fouraid] = 1.0
                    grayimg[int(sixpoint_center[fouraid][1])][int(sixpoint_center[fouraid][0])] = 255.0

            # 不规则图形只取外边框
            for idx, dic in enumerate(barriermorepoint_list):
                # print(idx, dic)
                for k, v in dic.items():
                    if BARRIER_LENGTH_AREA[0] <= k <= BARRIER_LENGTH_AREA[1]:  # 如果障碍在范围内
                        # 寻找区域
                        moreaid = 0
                        moredistance = 40000
                        for i, aid in enumerate(area_idx_list):
                            tmp_distance = math.sqrt( (v[0]-sixpoint_center[aid][0])**2 + (v[1]-sixpoint_center[aid][1])**2 )
                            if tmp_distance < moredistance:
                                moreaid = aid
                                moredistance = tmp_distance
                        if BARRIER_DISTANCE[0] <= moredistance <= BARRIER_DISTANCE[1]:
                            danger_state[moreaid] = 1.0
                            grayimg[int(sixpoint_center[moreaid][1])][int(sixpoint_center[moreaid][0])] = 255.0

        # print(danger_state)
        # cv2.imshow("window_main", grayimg)
        # cv2.moveWindow("window_main", 0, 520)
        # if cv2.waitKey(1) & 0xFF == ord('a'):
        #     print(cv2.waitKey(1))

        grayimg = grayimg + NUMPY_VALUE
        # area_state = np.array([danger_state, sixpoint_center, tri_center], dtype='object')
        area_state = [danger_state, sixpoint_center, tri_center]

        return grayimg, area_state

    # run()是Process类专门留出来用于重写的接口函数
    def run(self):
        while True:
            img = self.grab_screen(region=SUPERHEXAGON_WINDOW)
            img, area = self.self_image_conversion(img)
            # print(img.shape)

            #     break
            img = img[np.newaxis, :]  #(C,H,W) 3D卷积核(N,C,D,H,W)
            img = img[np.newaxis, :]
            torchimg = torch.as_tensor(img, dtype=torch.float32).to(CPUDEVICE)

            # area = area[np.newaxis, :]
            # torcharea = torch.as_tensor(area, dtype=torch.float32).to(CPUDEVICE)
            # print(torchimg.shape)
            # print(time.time(), self.screen_index.value)  # DEBUG

            # 滚动数组
            # print("debug:", (self.screen_index.value+1) % MANAGER_LIST_LENGTH)
            self.screen_list[(self.screen_index.value+1) % MANAGER_LIST_LENGTH] = torchimg
            self.area_list[(self.screen_index.value+1) % MANAGER_LIST_LENGTH] = area
            self.screen_index.value += 1
            # time.sleep(SLEEP_SCREEN)

class AeyeGrabscreen(object):
    def __init__(self):
        super(AeyeGrabscreen, self).__init__()
        # 进程间共享变量
        manager = Manager()
        self.screen_index = manager.Value('i', MANAGER_LIST_LENGTH)  # 为防止数组越界
        self.screen_list = manager.list([0 for i in range(MANAGER_LIST_LENGTH)])
        self.area_list = manager.list([0 for i in range(MANAGER_LIST_LENGTH)])

        grabscreen_process = GrabScreenProcess(screen_index=self.screen_index, screen_list=self.screen_list, area_list=self.area_list)
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

        area = self.area_list[x % MANAGER_LIST_LENGTH]
        # area = torch.as_tensor(area, dtype=torch.float32).to(DEVICE)
        # print(state.shape, x)
        return state, area



if __name__ == '__main__':
    # 单图片获取测试
    aeyegrabscreen = AeyeGrabscreen()
    while True:
        state, area = aeyegrabscreen.getstate()
        # time.sleep(0.2)
        # next_state = aeyegrabscreen.getstate()
        print(state)
        print(state.shape, area)


    # 差值图片获取测试
    # screen = AeyeGrabscreen()
    # last_screen = screen.getstate()
    # current_screen = screen.getstate()
    # next_state = current_screen - last_screen
    # while True:
    #     state = next_state.to(CPUDEVICE)
    #     state = np.array(state[0][0])
    #     # time.sleep(0.2)
    #     # next_state = aeyegrabscreen.getstate()
    #     # print(state)
    #     # print(state.shape)
    #
    #     cv2.imshow("window_main", state)
    #     cv2.moveWindow("window_main", 800, 340)
    #     if cv2.waitKey(1) & 0xFF == ord('a'):
    #         print(cv2.waitKey(1))
    #
    #     last_screen = current_screen
    #     current_screen = screen.getstate()
    #     next_state = current_screen - last_screen


    # while True:
    #     print(screen_index)
    #     cv2.imshow("window_main", screen_list[0])
    #     cv2.moveWindow("window_main", 800, 340)
    #     if cv2.waitKey(1) & 0xFF == ord('a'):
    #         print(cv2.waitKey(1))
    #         break
