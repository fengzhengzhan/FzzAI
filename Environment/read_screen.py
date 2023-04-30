# -*- coding: utf-8 -*-
import win32gui

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
