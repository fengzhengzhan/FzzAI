from dependencies import *


class ChangeEnv:
    """
    Change the environment status, make the program continuous.
    """

    def travelProcess(self):
        name_process = []

        def __enum_windows(hwnd, lParam):
            if win32gui.IsWindow(hwnd) and win32gui.IsWindowEnabled(hwnd) and win32gui.IsWindowVisible(hwnd):
                name_process.append({hwnd: win32gui.GetWindowText(hwnd)})

        win32gui.EnumWindows(__enum_windows, 0)
        # for np in name_process:
        #     print(np)
        return name_process

    def toppingProcess(self, name_process, x, y, width, height, has_nosize=False, has_nomove=False):
        # 0, 0, 1280, 720,
        hwnd = win32gui.FindWindow(None, name_process)
        win32gui.ShowWindow(hwnd, win32con.SW_SHOWNORMAL)  # 置顶窗口
        # Reference  https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setwindowpos
        uFlags = 0
        if has_nosize:  # Retains the current size (ignores the width and height parameters).
            uFlags = uFlags | win32con.SWP_NOSIZE
        if has_nomove:
            uFlags = uFlags | win32con.SWP_NOMOVE
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, x, y, width, height, uFlags)
        win32gui.SetForegroundWindow(hwnd)
        win32gui.SetActiveWindow(hwnd)

    def activeProcess(self, name_process):
        hwnd = win32gui.FindWindow(None, name_process)
        shell = win32com.client.Dispatch("WScript.Shell")  # 激活窗口
        shell.SendKeys('%')  # 发送Alt按键
        win32gui.SetForegroundWindow(hwnd)
        win32gui.SetActiveWindow(hwnd)


if __name__ == '__main__':
    reset = ChangeEnv()
    print(reset.travelProcess())
    reset.toppingProcess('无界面测试窗口.txt - 记事本', -10, 0, 1280, 720)

