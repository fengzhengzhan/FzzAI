import win32gui
import win32api
import win32process
import ctypes
import ctypes.wintypes
from config import *


class EscoreReadram():
    def __init__(self):
        hd = win32gui.FindWindow(None, GAME_HANDLE)
        pid = win32process.GetWindowThreadProcessId(hd)[1]
        self.process_handle = win32api.OpenProcess(0x1F0FFF, False, pid)
        self.kernal32 = ctypes.windll.LoadLibrary(r"C:\\Windows\\System32\\kernel32.dll")

    def get_score(self):
        base_address = 0x00691048  # 0x00691048  0x00694B00  0x00694B8C
        offset_address = ctypes.c_long()
        offset = 0x2988
        self.kernal32.ReadProcessMemory(int(self.process_handle), base_address, ctypes.byref(offset_address), 4, None)
        self.kernal32.ReadProcessMemory(int(self.process_handle), offset_address.value + offset, ctypes.byref(offset_address), 4, None)
        return offset_address.value


if __name__ == '__main__':
    score = EscoreReadram()
    while True:
        print(score.get_score())
        print(round(float(int(score.get_score()) / 60), 2))
