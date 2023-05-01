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

    def toppingProcess(self, name_process):
        hwnd = win32gui.FindWindow(None, name_process)
        win32gui.ShowWindow(hwnd, win32con.SW_SHOWNORMAL)  # 置顶窗口
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 1280, 720, win32con.SWP_NOSIZE)
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
    reset.toppingProcess('Steam')
