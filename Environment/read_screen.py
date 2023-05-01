# -*- coding: utf-8 -*-
from dependencies import *


class ReadScreen:
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

    def converScreen(self):
        return self.gainScreen()


class ProcessScreen(Process):
    """
    在多进程中使用manager传递信息
    Use Manager communicate information.
    """

    def __init__(self, channel_index, channel_data, channel_length):
        super(ProcessScreen, self).__init__()
        self.channel_index = channel_index
        self.channel_data = channel_data
        self.channel_length = channel_length

    def run(self):
        rs = ReadScreen()
        while True:
            # 多进程获取屏幕截图
            data = rs.converScreen()
            # 滚动Manager数组
            self.channel_data[(self.channel_index.value + 1) % self.channel_length] = data
            self.channel_index.value += 1


class ManagerScreen:
    def __init__(self, channel_length=64):
        super(ManagerScreen, self).__init__()
        self.__manager = Manager()
        self.channel_index = self.__manager.Value('i', channel_length)
        self.channel_data = self.__manager.list([0 for _ in range(channel_length)])
        self.channel_length = channel_length

        self.screen_process = ProcessScreen(self.channel_index, self.channel_data, self.channel_length)
        self.screen_process.start()
        # self.screen_process.join()

    def releaseResources(self):
        if self.screen_process.is_alive():
            self.screen_process.terminate()
            self.screen_process.join()
        self.__manager.shutdown()

    def gainTransData(self, data_count=1):
        index = self.channel_index.value
        data = []
        for count in range(data_count - 1, -1, -1):
            data.append(self.channel_data[(index - count) % self.channel_length])
        return data


if __name__ == '__main__':
    main_window = (0, 0, 800, 500)
    # blood_window = (222, 95, 530, 120)
    rs = ReadScreen(main_window)
    while True:
        # 测试代码 查看窗口位置
        screen_grey = cv2.cvtColor(rs.gainScreen(), cv2.COLOR_BGRA2BGR)
        cv2.imshow("window_main", screen_grey)
        cv2.moveWindow("window_main", 1000, 600)
        if cv2.waitKey(1) & 0xFF == ord('a'):
            print(cv2.waitKey(1))
            break
    cv2.waitKey()  # 视频结束后，按任意键退出
    cv2.destroyAllWindows()

    # 测试manager
    transport_manager = ManagerScreen()
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

