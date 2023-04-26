# -*- coding: utf-8 -*-
from dependencies import *


class ReadScreen:
    def __init__(self, region=None):
        # 得到图片范围
        if region:
            self.left, self.top, x2, y2 = region
            self.width = x2 - self.left + 1
            self.height = y2 - self.top + 1
        else:
            self.width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
            self.height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
            self.left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
            self.top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)
            # print(width, height, left, top)

        # 创建句柄，获取图片
        self.hwin = win32gui.GetDesktopWindow()
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

    def gainScreen(self, region=None):
        self.memdc.SelectObject(self.bmp)
        self.memdc.BitBlt((0, 0), (self.width, self.height), self.srcdc, (self.left, self.top), win32con.SRCCOPY)

        self.signedIntsArray = self.bmp.GetBitmapBits(True)
        self.img = np.frombuffer(self.signedIntsArray, dtype='uint8')
        self.img.shape = (self.height, self.width, 4)

        return self.img
        # return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)


if __name__ == '__main__':
    # ReadScreen().gainScreen()
    # left, top, x2, y2
    main_window = (60, 99, 1219, 610)
    # blood_window = (222, 95, 530, 120)
    rs = ReadScreen(main_window)
    while True:
        # 测试代码 查看窗口位置
        second_screen_grey = cv2.cvtColor(rs.gainScreen(), cv2.COLOR_BGRA2BGR)
        cv2.imshow("window_main", second_screen_grey)
        cv2.moveWindow("window_main", 700, 540)
        if cv2.waitKey(1) & 0xFF == ord('a'):
            print(cv2.waitKey(1))
            break
    cv2.waitKey()  # 视频结束后，按任意键退出
    cv2.destroyAllWindows()
