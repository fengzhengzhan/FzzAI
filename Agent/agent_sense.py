from dependencies import *


class Sense(object):
    """
    输入信息：项目以输入的感官信息作为分析执行依据。
    Input information: The project uses the input sensory information as the basis for analysis and execution.
    """

    def __init__(self):
        super(Sense, self).__init__()

    def __del__(self):
        pass

    def visionInit(self, vision_list: list):
        pass

    def visionScreen(self):
        # 获取
        hwin = win32gui.GetDesktopWindow()

        # 得到图片范围
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

    def vision(self):
        pass

    def hearing(self):
        pass

    def smell(self):
        pass

    def taste(self):
        pass

    def touch(self):
        pass

    def perception(self):
        pass
