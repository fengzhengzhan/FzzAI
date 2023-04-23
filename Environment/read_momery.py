from dependencies import *


class ReadMemory:
    def __init__(self, game_window_title):
        window_handle = win32gui.FindWindow(None, game_window_title)
        process_id = win32process.GetWindowThreadProcessId(window_handle)[1]
        self.process_handle = win32api.OpenProcess(0x1F0FFF, False, process_id)
        self.kernal32 = ctypes.windll.LoadLibrary(r"C:\\Windows\\System32\\kernel32.dll")

    def gainValue(self, base_address, offset_address):
        offset = ctypes.c_long()
        self.kernal32.ReadProcessMemory(int(self.process_handle), base_address, ctypes.byref(offset), 4, None)
        self.kernal32.ReadProcessMemory(
            int(self.process_handle), offset.value + offset_address, ctypes.byref(offset), 4, None
        )
        return offset.value


if __name__ == '__main__':
    score = ReadMemory("Super Hexagon")
    base_address = 0x00691048  # 0x00691048  0x00694B00  0x00694B8C
    offset_address = 0x2988
    while True:
        print(score.gainValue(base_address, offset_address))
        print(round(float(int(score.gainValue(base_address, offset_address)) / 60), 2))
