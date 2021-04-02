import win32gui
import win32con
import time
#
# hwnd_title = []
#
# def get_all_hwnd(hwnd, mouse):
#     if (win32gui.IsWindow(hwnd) and win32gui.IsWindowEnabled(hwnd) and win32gui.IsWindowVisible(hwnd)):
#         hwnd_title.append({hwnd: win32gui.GetWindowText(hwnd)})
#
#
# win32gui.EnumWindows(get_all_hwnd, 0)
# for h in hwnd_title:
#     print(h)
def handld_top():
    hwnd = win32gui.FindWindow(None, 'Hollow Knight')
    win32gui.ShowWindow(hwnd, win32con.SW_SHOWNORMAL)
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 1030, 560, 0)
    win32gui.SetForegroundWindow(hwnd)
    win32gui.SetActiveWindow(hwnd)

if __name__ == '__main__':
    handld_top()