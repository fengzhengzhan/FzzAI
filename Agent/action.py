import time

from dependencies import *


######################
# Structure
######################
class StructKeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),  # 虚拟键码
        ("wScan", ctypes.c_ushort),  # 硬件扫描码
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),  # 按键事件时间戳，0当前系统时间作为时间戳
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class StructMouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class StructHardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort),
    ]


class UnionKeyInput(ctypes.Union):
    _fields_ = [
        ("ki", StructKeyBdInput),
        ("mi", StructMouseInput),
        ("hi", StructHardwareInput),
    ]


class StructInput(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("ii", UnionKeyInput),
    ]


class StructPoint(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_ulong),
        ("y", ctypes.c_ulong)
    ]


######################
# Action
######################
class ActionKeyboard:
    """
    键盘模拟
    KeyBoard Simulate
    """

    def __init__(self, short_during_time=0.05, long_during_time=0.2):
        # 虚拟键码
        # self.__DWFLAGS_PRESSKEY = 0x0000
        # self.__DWFLAGS_RELEASEKEY = 0x0000 | 0x0002
        # 硬件扫描码
        self.__DWFLAGS_PRESSKEY = 0x0008
        self.__DWFLAGS_RELEASEKEY = 0x0008 | 0x0002

        self.__short_during_time = short_during_time
        self.__long_during_time = long_during_time

        # 虚拟键码：104键横向排列
        # self.__scancode = {
        #     'ESC': 0x1B, 'F1': 0x70, 'F2': 0x71, 'F3': 0x72, 'F4': 0x73, 'F5': 0x74, 'F6': 0x75,
        #     'F7': 0x76, 'F8': 0x77, 'F9': 0x78, 'F10': 0x79, 'F11': 0x7A, 'F12': 0x7B,
        #     '`': 0xC0, '1': 0x31, '2': 0x32, '3': 0x33, '4': 0x34, '5': 0x35, '6': 0x36, '7': 0x37,
        #     '8': 0x38, '9': 0x39, '0': 0x30, '-': 0xBD, '=': 0xBB, 'Backspace': 0x08,
        #     'Tab': 0x09, 'Q': 0x51, 'W': 0x57, 'E': 0x45, 'R': 0x52, 'T': 0x54, 'Y': 0x59, 'U': 0x55,
        #     'I': 0x49, 'O': 0x4F, 'P': 0x50, '[': 0xDB, ']': 0xDD, '\\': 0xDC,
        #     'CapsLock': 0x14, 'A': 0x41, 'S': 0x53, 'D': 0x44, 'F': 0x46, 'G': 0x47, 'H': 0x48, 'J': 0x4A,
        #     'K': 0x4B, 'L': 0x4C, ';': 0xBA, '\'': 0xDE, 'Enter': 0x0D,
        #     'LeftShift': 0xA0, 'Z': 0x5A, 'X': 0x58, 'C': 0x43, 'V': 0x56, 'B': 0x42, 'N': 0x4E,
        #     'M': 0x4D, ',': 0xBC, '.': 0xBE, '/': 0xBF, 'RightShift': 0xA1,
        #     'LeftCtrl': 0xA2, 'LeftWin': 0x5B, 'LeftAlt': 0xA4, 'Space': 0x20, 'RightAlt': 0xA5,
        #     'RightWin': 0x5C, 'RightCtrl': 0xA3, 'Fn': 0x00,
        #     'PrintScreen': 0x2C, 'ScrollLock': 0x91, 'PauseBreak': 0x13,
        #     'Insert': 0x2D, 'Home': 0x24, 'PageUp': 0x21,
        #     'Delete': 0x2E, 'End': 0x23, 'PageDown': 0x22,
        #     'Up': 0x26, 'Left': 0x25, 'Down': 0x28, 'Right': 0x27,
        #     'NumLock': 0x90, 'Num/': 0x6F, 'Num*': 0x6A, 'Num-': 0x6D, 'Num7': 0x67, 'Num8': 0x68, 'Num9': 0x69,
        #     'Num4': 0x64, 'Num5': 0x65, 'Num6': 0x66, 'Num+': 0x6B, 'Num1': 0x61, 'Num2': 0x62, 'Num3': 0x63,
        #     'Num0': 0x60, 'Num.': 0x6E, 'NumEnter': 0x0D,
        # }

        # 硬件扫描码，扫描码基于 IBM PC 兼容键盘
        self.__scancode = {
            'ESC': 0x01, 'F1': 0x3B, 'F2': 0x3C, 'F3': 0x3D, 'F4': 0x3E, 'F5': 0x3F, 'F6': 0x40,
            'F7': 0x41, 'F8': 0x42, 'F9': 0x43, 'F10': 0x44, 'F11': 0x57, 'F12': 0x58,
            '`': 0x29, '1': 0x02, '2': 0x03, '3': 0x04, '4': 0x05, '5': 0x06, '6': 0x07, '7': 0x08,
            '8': 0x09, '9': 0x0A, '0': 0x0B, '-': 0x0C, '=': 0x0D, 'Backspace': 0x0E,
            'Tab': 0x0F, 'Q': 0x10, 'W': 0x11, 'E': 0x12, 'R': 0x13, 'T': 0x14, 'Y': 0x15, 'U': 0x16,
            'I': 0x17, 'O': 0x18, 'P': 0x19, '[': 0x1A, ']': 0x1B, '\\': 0x2B,
            'CapsLock': 0x3A, 'A': 0x1E, 'S': 0x1F, 'D': 0x20, 'F': 0x21, 'G': 0x22, 'H': 0x23, 'J': 0x24,
            'K': 0x25, 'L': 0x26, ';': 0x27, '\'': 0x28, 'Enter': 0x1C,
            'LeftShift': 0x2A, 'Z': 0x2C, 'X': 0x2D, 'C': 0x2E, 'V': 0x2F, 'B': 0x30, 'N': 0x31,
            'M': 0x32, ',': 0x33, '.': 0x34, '/': 0x35, 'RightShift': 0x36,
            'LeftCtrl': 0x1D, 'LeftWin': 0xE05B, 'LeftAlt': 0x38, 'Space': 0x39, 'RightAlt': 0xE038,
            'RightWin': 0xE05C, 'RightCtrl': 0xE01D,  # 'Fn': None,
            'PrintScreen': 0xE037, 'ScrollLock': 0x46, 'PauseBreak': 0xE11D45,
            'Insert': 0xE052, 'Home': 0xE047, 'PageUp': 0xE049,
            'Delete': 0xE053, 'End': 0xE04F, 'PageDown': 0xE051,
            'Up': 0xE048, 'Left': 0xE04B, 'Down': 0xE050, 'Right': 0xE04D,
            'NumLock': 0x45, 'Num/': 0xE035, 'Num*': 0x37, 'Num-': 0x4A, 'Num7': 0x47, 'Num8': 0x48, 'Num9': 0x49,
            'Num4': 0x4B, 'Num5': 0x4C, 'Num6': 0x4D, 'Num+': 0x4E, 'Num1': 0x4F, 'Num2': 0x50, 'Num3': 0x51,
            'Num0': 0x52, 'Num.': 0x53, 'NumEnter': 0xE01C,
        }

    def __operateKey(self, str_key, dwflags):
        hex_scancode = self.__scancode[str_key]
        uki = UnionKeyInput()
        # uki.ki = StructKeyBdInput(hex_scancode, 0, dwflags, 0, ctypes.pointer(ctypes.c_ulong(0)))  # 虚拟键码
        uki.ki = StructKeyBdInput(0, hex_scancode, dwflags, 0, ctypes.pointer(ctypes.c_ulong(0)))  # 硬件扫描码
        si = StructInput(ctypes.c_ulong(1), uki)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(si), ctypes.sizeof(si))

    def __finishKey(self, str_key, during_time):
        self.__operateKey(str_key, self.__DWFLAGS_PRESSKEY)
        time.sleep(during_time)
        self.__operateKey(str_key, self.__DWFLAGS_RELEASEKEY)

    def shortKey(self, str_key):
        self.__finishKey(str_key, self.__short_during_time)

    def longKey(self, str_key):
        self.__finishKey(str_key, self.__long_during_time)

    def duringKey(self, str_key, during_time):
        self.__finishKey(str_key, during_time)

    def customTimeKey(self, str_key, during_time):
        self.__finishKey(str_key, during_time)

    def doubleKey(self, str_key):
        self.shortKey(str_key)
        time.sleep(self.__short_during_time)
        self.shortKey(str_key)


class ActionMouse:
    """
    鼠标模拟
    Mouse Simulate
    """

    def __init__(self, time_click_interval=0.01):
        self.__MOUSEEVENTF_MOVE = 0x0001
        self.__MOUSEEVENTF_LEFTDOWN = 0x0002
        self.__MOUSEEVENTF_LEFTUP = 0x0004
        self.__MOUSEEVENTF_MIDDLEDOWN = 0x0020
        self.__MOUSEEVENTF_MIDDLEUP = 0x0040
        self.__MOUSEEVENTF_RIGHTDOWN = 0x0008
        self.__MOUSEEVENTF_RIGHTUP = 0x0010
        self.__MOUSEEVENTF_WHEEL = 0x0800
        self.__MOUSEEVENTF_ABSOLUTE = 0x8000

        self.time_click_interval = time_click_interval

        self.__map_mouse = {'left': 'l', 'middle': 'm', 'right': 'r'}

    def getScreenSize(self):
        # 获取当前屏幕分辨率。
        get_system_metrics = ctypes.windll.user32.GetSystemMetrics
        return (get_system_metrics(0), get_system_metrics(1))

    def getPosition(self):
        # 获取当前鼠标位置。返回值为一个元组,以(x,y)形式表示。
        position = StructPoint()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(position))
        return (int(position.x), int(position.y))

    def __generateRandlist(self, n, target_sum):
        """
        生成一个随机数组，其中中间值较大，两头较小
        参数：
            n：数组元素个数
            target_sum：数组的和
        """
        # 生成一个初始的随机数组
        rand_array = [random.random() for _ in range(n)]

        # 为了让中间值较大，两头较小，我们将数组的元素乘以一个二次函数
        xs = [i / (n - 1) for i in range(n)]
        ys = [0.1 + 0.9 * (1 - (2 * x - 1) ** 2) for x in xs]  # 保证首尾元素不为0
        adjusted_randarray = [rand_array[i] * ys[i] for i in range(n)]

        # 确保数组之和为 target_sum
        sum_randarray = sum(adjusted_randarray)
        final_array = [round(x * target_sum / sum_randarray, 3) for x in adjusted_randarray]

        return final_array

    def __move(self, x, y, dtime=None):
        # 平滑移动：将鼠标移动至指定位置(x, y)
        # 模拟移动策略：移动步数随机化；移动距离两头慢中间快；移动休眠时间两头慢中间快；移动总时间长度越长时间越长
        steps = random.randint(7, 11)  # 移动步数随机化
        x_pos, y_pos = self.getPosition()

        # 移动总时间长度越长时间越长
        x_dis, y_dis = x - x_pos, y - y_pos
        x_screen, y_screen = self.getScreenSize()
        if dtime == None:
            if (abs(x_dis) <= x_screen // 4) and (abs(y_dis) <= y_screen // 4):
                dtime = 0.02
            elif (abs(x_dis) <= x_screen // 2) and (abs(y_dis) <= y_screen // 2):
                dtime = 0.03
            else:
                dtime = 0.05

        # 移动距离两头慢中间快
        list_xsteps = self.__generateRandlist(steps, 1)
        list_ysteps = self.__generateRandlist(steps, 1)

        # 移动休眠时间两头慢中间快
        list_interval = self.__generateRandlist(steps, dtime)

        for i in range(0, steps):
            x_current = x_pos + x_dis * list_xsteps[i]
            y_current = y_pos + y_dis * list_ysteps[i]
            ctypes.windll.user32.SetCursorPos(int(x_current), int(y_current))  # 不产生一个鼠标事件
            x_pos, y_pos = x_current, y_current
            time.sleep(list_interval[i])
        ctypes.windll.user32.SetCursorPos(int(x), int(y))

    def moveAbsolute(self, tx, ty):
        self.__move(tx, ty)

    def moveReletive(self, dx, dy):
        x_pos, y_pos = self.getPosition()
        x_screen, y_screen = self.getScreenSize()
        tx = x_pos + dx
        ty = y_pos + dy

        # 将值限制在屏幕之内
        tx = max(0, min(tx, x_screen - 1))
        ty = max(0, min(ty, y_screen - 1))

        self.__move(tx, ty)

    def __operateMouse(self, dw_flags):
        mi = StructMouseInput(0, 0, 0, dw_flags, 0, ctypes.pointer(ctypes.c_ulong(0)))
        ii = UnionKeyInput(mi=mi)
        si = StructInput(ctypes.c_ulong(0), ii)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(si), ctypes.sizeof(si))

    def __mapKey(self, str_key):
        if str_key == self.__map_mouse['left']:
            dwflags_down = self.__MOUSEEVENTF_LEFTDOWN
            dwflags_up = self.__MOUSEEVENTF_LEFTUP
        elif str_key == self.__map_mouse['middle']:
            dwflags_down = self.__MOUSEEVENTF_MIDDLEDOWN
            dwflags_up = self.__MOUSEEVENTF_MIDDLEUP
        elif str_key == self.__map_mouse['right']:
            dwflags_down = self.__MOUSEEVENTF_RIGHTDOWN
            dwflags_up = self.__MOUSEEVENTF_RIGHTUP
        else:
            raise ValueError("Error: Mouse click value error!")

        return (dwflags_down, dwflags_up)

    def __finishMouse(self, str_key, during_time):
        dwflags_down, dwflags_up = self.__mapKey(str_key)

        self.__operateMouse(dwflags_down)
        time.sleep(during_time)
        self.__operateMouse(dwflags_up)

    def click(self, str_key):
        self.__finishMouse(str_key, 0.05)

    def duringClick(self, str_key, during_time):
        self.__finishMouse(str_key, during_time)

    def doubleClick(self, str_key):
        self.click(str_key)
        time.sleep(self.time_click_interval)
        self.click(str_key)

    def drag(self, str_key, sx, sy, ex, ey):
        dwflags_down, dwflags_up = self.__mapKey(str_key)
        self.__move(sx, sy)
        self.__operateMouse(dwflags_down)
        self.__move(ex, ey)
        self.__operateMouse(dwflags_up)


# Actuals Functions


# use keyboard
# I = 0x17
# J = 0x24
# K = 0x25
# L = 0x26
#
# SPACE = 0x39
#
# Q = 0x10
# W = 0x11
# E = 0x12
# R = 0x13
#
# ENTER = 0x1C
# T = 0x14
# ESC = 0x01
#
#
# # 移动
# def i_up():
#     PressKey(I)
#     time.sleep(0.15)
#     ReleaseKey(I)
#     # time.sleep(0.2)
#
#
# def j_left():
#     PressKey(J)
#     time.sleep(0.1)
#     ReleaseKey(J)
#     PressKey(R)
#     time.sleep(0.05)
#     ReleaseKey(R)
#     # time.sleep(0.2)
#
#
# # def j_left():
# #     PressKey(J)
# #     time.sleep(0.1)
# #     ReleaseKey(J)
# #     # PressKey(R)
# #     # time.sleep(0.05)
# #     # ReleaseKey(R)
# #     # time.sleep(0.2)
#
# def k_down():
#     PressKey(K)
#     time.sleep(0.15)
#     ReleaseKey(K)
#     # time.sleep(0.2)
#
#
# def l_right():
#     PressKey(L)
#     time.sleep(0.1)
#     ReleaseKey(L)
#     PressKey(R)
#     time.sleep(0.05)
#     ReleaseKey(R)
#     # time.sleep(0.2)
#
#
# # def l_right():
# #     PressKey(L)
# #     time.sleep(0.1)
# #     ReleaseKey(L)
# #     # PressKey(R)
# #     # time.sleep(0.05)
# #     # ReleaseKey(R)
# #     # time.sleep(0.2)
#
# def i_attack():
#     PressKey(I)
#     # time.sleep(0.05)
#     PressKey(R)
#     time.sleep(0.05)
#     ReleaseKey(I)
#     ReleaseKey(R)
#
#
# def k_attack():
#     PressKey(K)
#     # time.sleep(0.05)
#     PressKey(R)
#     time.sleep(0.05)
#     ReleaseKey(K)
#     ReleaseKey(R)
#
#
# # 跳跃
# def space_long_up():
#     PressKey(SPACE)
#     time.sleep(0.5)
#     ReleaseKey(SPACE)
#     # time.sleep(0.2)
#
#
# def space_short_up():
#     PressKey(SPACE)
#     time.sleep(0.15)
#     ReleaseKey(SPACE)
#     # time.sleep(0.2)
#
#
# # 技能
# def q_short_use():
#     # TODO test
#     PressKey(Q)
#     time.sleep(0.05)
#     ReleaseKey(Q)
#     # time.sleep(0.2)
#
#
# def q_long_use():
#     PressKey(Q)
#     time.sleep(1.4)
#     ReleaseKey(Q)
#     # time.sleep(0.2)
#
#
# # 攻击
# def r_short_attack():
#     PressKey(R)
#     time.sleep(0.05)
#     ReleaseKey(R)
#     # time.sleep(0.2)
#
#
# # def r_long_attack():
# #     PressKey(R)
# #     time.sleep(1.6)
# #     ReleaseKey(R)
# #     # time.sleep(0.2)
#
# # 运动
# def w_long_run():
#     # TODO test
#     PressKey(W)
#     time.sleep(0.5)
#     ReleaseKey(W)
#     # time.sleep(0.2)
#
#
# def e_run():
#     PressKey(E)
#     time.sleep(0.05)
#     ReleaseKey(E)
#     # time.sleep(0.2)
#
#
# # 技能
# def iq_boom():
#     PressKey(I)
#     PressKey(Q)
#     # time.sleep(0.05)
#     ReleaseKey(Q)
#     ReleaseKey(I)
#
#
# def jq_boom():
#     PressKey(J)
#     time.sleep(0.02)
#     PressKey(Q)
#     # time.sleep(0.05)
#     ReleaseKey(Q)
#     ReleaseKey(J)
#
#
# def kq_doom():
#     PressKey(K)
#     PressKey(Q)
#     ReleaseKey(Q)
#     ReleaseKey(K)
#
#
# def lq_boom():
#     PressKey(L)
#     time.sleep(0.02)
#     PressKey(Q)
#     # time.sleep(0.05)
#     ReleaseKey(Q)
#     ReleaseKey(L)
#
#
# def t_pause():
#     PressKey(T)
#     time.sleep(0.05)
#     ReleaseKey(T)
#     # time.sleep(0.2)
#
#
# def esc_quit():
#     PressKey(ESC)
#     time.sleep(0.05)
#     ReleaseKey(ESC)
#     # time.sleep(0.2)
#
#
# # 竞技场
# # def init_start():
# #     print("[-] Knight dead,restart...")
# #     # 椅子到通道
# #     time.sleep(3)
# #     PressKey(J)
# #     time.sleep(0.05)
# #     ReleaseKey(J)
# #     time.sleep(2.5)
# #     PressKey(W)
# #     time.sleep(1.3)
# #     ReleaseKey(W)
# #     time.sleep(1.9)
# #     PressKey(W)
# #     time.sleep(0.05)
# #     ReleaseKey(W)
# #     time.sleep(2)
# #
# #     # 通道
# #     PressKey(SPACE)
# #     time.sleep(0.5)
# #     ReleaseKey(SPACE)
# #     time.sleep(0.05)
# #
# #     PressKey(SPACE)
# #     PressKey(J)
# #     time.sleep(0.5)
# #     ReleaseKey(SPACE)
# #
# #     time.sleep(0.05)
# #     PressKey(SPACE)
# #     time.sleep(0.05)
# #     ReleaseKey(SPACE)
# #     time.sleep(0.05)
# #
# #     PressKey(SPACE)
# #     time.sleep(0.05)
# #     ReleaseKey(SPACE)
# #     time.sleep(0.05)
# #
# #     # PressKey(SPACE)
# #     # time.sleep(0.05)
# #     # ReleaseKey(SPACE)
# #     PressKey(SPACE)
# #     time.sleep(0.5)
# #     ReleaseKey(SPACE)
# #     ReleaseKey(J)
# #
# #     # 左拿徽章
# #     time.sleep(4.5)
# #     PressKey(J)
# #     time.sleep(0.4)
# #     ReleaseKey(J)
# #
# #     # 通行证
# #     PressKey(I)
# #     time.sleep(0.05)
# #     ReleaseKey(I)
# #     time.sleep(1.5)
# #     PressKey(J)
# #     time.sleep(0.05)
# #     ReleaseKey(J)
# #     time.sleep(0.05)
# #     PressKey(ENTER)
# #     time.sleep(0.05)
# #     ReleaseKey(ENTER)
# #
# #     # 右进角斗场
# #     time.sleep(2.5)
# #     PressKey(L)
# #     time.sleep(0.05)
# #     ReleaseKey(L)
# #     time.sleep(0.05)
# #     PressKey(W)
# #     time.sleep(1.3)
# #     ReleaseKey(W)
# #     time.sleep(7.5)
# #     PressKey(W)
# #     time.sleep(0.05)
# #     ReleaseKey(W)
# #     print("[+] Enter start...")
#
# # 神居
# def init_init():
#     time.sleep(3)
#     PressKey(I)
#     time.sleep(0.05)
#     ReleaseKey(I)
#     time.sleep(1)
#     PressKey(SPACE)
#     time.sleep(0.05)
#     ReleaseKey(SPACE)
#     time.sleep(6)
#     print("[+] Enter start...")
#
#
# # 神居
# def init_start():
#     print("[-] Knight dead,restart...")
#     # 起身
#     PressKey(SPACE)
#     time.sleep(0.05)
#     ReleaseKey(SPACE)
#     time.sleep(2)
#
#     PressKey(SPACE)
#     time.sleep(0.05)
#     ReleaseKey(SPACE)
#     time.sleep(2)
#
#     # 选择
#     PressKey(I)
#     time.sleep(0.05)
#     ReleaseKey(I)
#     time.sleep(1)
#     PressKey(SPACE)
#     time.sleep(0.05)
#     ReleaseKey(SPACE)
#     time.sleep(6)
#     print("[+] Enter start...")


class Action:
    def __init__(self, keyboard=False, mouse=False, xbox=False, video=False, audio=False):
        self.keyboard = ActionKeyboard()
        self.mouse = ActionMouse()


if __name__ == "__main__":
    # Unit Test

    # action = Action()
    # tx, ty = action.mouse.getPosition()
    # print(tx, ty)
    # print(action.mouse.getPosition())
    # print(action.mouse.getScreenSize())
    # while True:
    #     time.sleep(2)
    # action.mouse.moveAbsolute(1000, 500)
    # action.mouse.moveAbsolute(200, 200)
    # action.mouse.moveReletive(1000, 500)
    # action.mouse.click(action.mouse.key['right'])
    # action.mouse.drag(action.mouse.key['left'], 500, 300, 800, 500)

    # action = Action()
    # for k, v in action.keyboard.key.items():
    #     time.sleep(0.2)
    #     print(k, end=" ")
    #     action.keyboard.shortKey(v)
    #

    action = Action()
    while True:
        time.sleep(0.3)
        # action.keyboard.duringKey('E', 0.5)
        action.mouse.duringClick('r', 0.5)

    # eeeeeeeeeeeeeeeeeeeeeeeeeeeeeee

    # while True:  # 16种操作全部测试
    # i_up()
    # j_left()
    # time.sleep(1)
    # k_down()
    # l_right()
    # space_long_up()
    # space_short_up()
    # q_short_use()
    # q_long_use()
    # r_short_attack()
    # r_long_attack()
    # w_long_run()
    # e_run()
    # iq_boom()
    # kq_doom()
    # jq_boom()
    # lq_boom()
    # time.sleep(1)
    # init_start()
