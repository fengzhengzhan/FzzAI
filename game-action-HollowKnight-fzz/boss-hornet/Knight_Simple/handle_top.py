import win32gui
import win32con
import time
#
#　将窗口置顶，防止意外情况，延长训练时间
def handld_top():
    hwnd = win32gui.FindWindow(None, 'Hollow Knight')
    win32gui.ShowWindow(hwnd, win32con.SW_SHOWNORMAL)
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 1030, 560, 0)
    win32gui.SetForegroundWindow(hwnd)
    win32gui.SetActiveWindow(hwnd)

if __name__ == '__main__':
    handld_top()