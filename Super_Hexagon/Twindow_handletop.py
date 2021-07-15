import win32gui, win32com.client
from config import *
import win32con
import time

# hwnd_title = []
#
# def get_all_hwnd(hwnd, mouse):
#     if (win32gui.IsWindow(hwnd) and win32gui.IsWindowEnabled(hwnd) and win32gui.IsWindowVisible(hwnd)):
#         hwnd_title.append({hwnd: win32gui.GetWindowText(hwnd)})

# win32gui.EnumWindows(get_all_hwnd, 0)
# for h in hwnd_title:
#     print(h)

def handle_top():
    try:
        hwnd = win32gui.FindWindow(None, GAME_HANDLE)
        # win32gui.ShowWindow(hwnd, win32con.SW_SHOWNORMAL)
        # win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 1030, 560, 0)
        shell = win32com.client.Dispatch("WScript.Shell")
        shell.SendKeys('%')
        win32gui.SetForegroundWindow(hwnd)
        win32gui.SetActiveWindow(hwnd)
    except Exception as e:
        print("[*] Twindow_handletop error:", e)

if __name__ == '__main__':
    handle_top()