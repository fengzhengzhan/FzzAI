# -*- coding: utf-8 -*-
from dependencies import *


class GrabScreen:
    def __init__(self, region=(0, 0, 1280, 720), name_process=None):
        # 得到图片范围
        self.left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        self.top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)
        self.width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        self.height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        # region (x, y, width, height)
        if region:
            self.left, self.top, self.width, self.height = region
        # print(width, height, left, top)

        # region (x1, y1, x2, y2)
        # self.left, self.top, x2, y2 = region
        # self.width = x2 - self.left + 1
        # self.height = y2 - self.top + 1

        # 创建句柄，获取图片
        self.hwin = win32gui.GetDesktopWindow()
        if name_process:
            self.hwin = win32gui.FindWindow(None, name_process)  # 可得到指定程序的窗口句柄
        self.hwindc = win32gui.GetWindowDC(self.hwin)
        self.srcdc = win32ui.CreateDCFromHandle(self.hwindc)
        self.memdc = self.srcdc.CreateCompatibleDC()
        self.bmp = win32ui.CreateBitmap()
        self.bmp.CreateCompatibleBitmap(self.srcdc, self.width, self.height)

    def __del__(self):
        self.srcdc.DeleteDC()
        self.memdc.DeleteDC()
        win32gui.ReleaseDC(self.hwin, self.hwindc)
        win32gui.DeleteObject(self.bmp.GetHandle())

    def resizeRegion(self, region):
        # 调整截图范围
        self.left, self.top, self.width, self.height = region
        self.bmp.CreateCompatibleBitmap(self.srcdc, self.width, self.height)

    def gainScreen(self):
        # 得到屏幕截图
        self.memdc.SelectObject(self.bmp)
        self.memdc.BitBlt((0, 0), (self.width, self.height), self.srcdc, (self.left, self.top), win32con.SRCCOPY)

        signedIntsArray = self.bmp.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (self.height, self.width, 4)

        return img
        # return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)


class ProcessReadScreen(Process):
    """
    在多进程中使用manager传递信息
    Use Manager communicate information.
    """

    def __init__(self):
        super(ProcessReadScreen, self).__init__()
        self.channel_length = None
        self.channel_data = None
        self.channel_index = None

    def setChannel(self, channel_index, channel_data, channel_length):
        self.channel_index = channel_index
        self.channel_data = channel_data
        self.channel_length = channel_length

    def converScreen(self, img):
        # 装饰类，转换屏幕截图图像
        return img

    def run(self):
        gs = GrabScreen()
        while True:
            # 多进程获取屏幕截图
            data = self.converScreen(gs.gainScreen())
            # 滚动Manager数组
            self.channel_data[(self.channel_index.value + 1) % self.channel_length] = data
            self.channel_index.value += 1


if __name__ == '__main__':
    # main_window = (0, 0, 800, 500)
    # # blood_window = (222, 95, 530, 120)
    # rs = GrabScreen(main_window)
    # while True:
    #     # 测试代码 查看窗口位置
    #     screen_grey = cv2.cvtColor(rs.gainScreen(), cv2.COLOR_BGRA2BGR)
    #     cv2.imshow("window_main", screen_grey)
    #     cv2.moveWindow("window_main", 1000, 600)
    #     if cv2.waitKey(1) & 0xFF == ord('a'):
    #         print(cv2.waitKey(1))
    #         break
    # cv2.waitKey()  # 视频结束后，按任意键退出
    # cv2.destroyAllWindows()

    # 测试manager
    transport_manager = TransportManager(ProcessReadScreen())
    time.sleep(3)

    # 并行性验证
    for i in range(100000):
        print(i)
        print(transport_manager.channel_index.value)

    # 截图正确性验证
    while True:
        # 测试代码 查看窗口位置
        screen_grey = cv2.cvtColor(transport_manager.gainTransData()[0], cv2.COLOR_BGRA2BGR)
        cv2.imshow("window_main", screen_grey)
        cv2.moveWindow("window_main", 1000, 600)
        if cv2.waitKey(1) & 0xFF == ord('a'):
            print(cv2.waitKey(1))
            break
    cv2.waitKey()  # 视频结束后，按任意键退出
    cv2.destroyAllWindows()

    transport_manager.releaseResources()
