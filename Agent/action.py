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

        # 虚拟键码
        # self.key = {'ESC': 0x1B, 'F1': 0x70, 'F2': 0x71, 'F3': 0x72, 'F4': 0x73, 'F5': 0x74, 'F6': 0x75,
        #             'F7': 0x76, 'F8': 0x77, 'F9': 0x78, 'F10': 0x79, 'F11': 0x7A, 'F12': 0x7B,
        #             '`': 0xC0, '0': 0x30,
        #
        #             'E': 0x45, 'W': 0x57,
        #
        #             'I': 0x49,
        #             'J': 0x24,
        #             'K': 0x25,
        #             'L': 0x26, }

        # 硬件扫描码
        self.key = {'E': 0x12, 'W': 0x11}

    def __operateKey(self, hex_key_code, dwflags):
        uki = UnionKeyInput()
        # uki.ki = StructKeyBdInput(hex_key_code, 0, dwflags, 0, ctypes.pointer(ctypes.c_ulong(0)))  # 虚拟键码
        uki.ki = StructKeyBdInput(0, hex_key_code, dwflags, 0, ctypes.pointer(ctypes.c_ulong(0)))  # 虚拟键码
        si = StructInput(ctypes.c_ulong(1), uki)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(si), ctypes.sizeof(si))

    def __finishKey(self, hex_key_code, during_time):
        self.__operateKey(hex_key_code, self.__DWFLAGS_PRESSKEY)
        time.sleep(during_time)
        self.__operateKey(hex_key_code, self.__DWFLAGS_RELEASEKEY)

    def shortKey(self, hex_key_code):
        self.__finishKey(hex_key_code, self.__short_during_time)

    def longKey(self, hex_key_code):
        self.__finishKey(hex_key_code, self.__long_during_time)

    def duringKey(self, hex_key_code, during_time):
        self.__finishKey(hex_key_code, during_time)

    def customTimeKey(self, hex_key_code, during_time):
        self.__finishKey(hex_key_code, during_time)

    def doubleKey(self, hex_key_code):
        self.shortKey(hex_key_code)
        time.sleep(self.__short_during_time)
        self.shortKey(hex_key_code)


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

        self.key = {'left': 'l', 'middle': 'm', 'right': 'r'}

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

    def move(self, dx, dy, dtime=None):
        # 平滑移动：将鼠标移动至指定位置(x, y)
        # 模拟移动策略：移动步数随机化；移动距离两头慢中间快；移动休眠时间两头慢中间快；移动总时间长度越长时间越长
        steps = random.randint(7, 11)  # 移动步数随机化
        x_pos, y_pos = self.getPosition()

        # 移动总时间长度越长时间越长
        x_dis, y_dis = dx - x_pos, dy - y_pos
        x_screen, y_screen = self.getScreenSize()
        if dtime == None:
            if (x_dis <= x_screen // 4) and (y_dis <= y_screen // 4):
                dtime = 0.02
            elif (x_dis <= x_screen // 2) and (y_dis <= y_screen // 2):
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
        ctypes.windll.user32.SetCursorPos(int(dx), int(dy))

    def __operateMouse(self, dw_flags):
        mi = StructMouseInput(0, 0, 0, dw_flags, 0, ctypes.pointer(ctypes.c_ulong(0)))
        ii = UnionKeyInput(mi=mi)
        si = StructInput(ctypes.c_ulong(0), ii)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(si), ctypes.sizeof(si))

    def __mapKey(self, hex_key_code):
        if hex_key_code == self.key['left']:
            dwflags_down = self.__MOUSEEVENTF_LEFTDOWN
            dwflags_up = self.__MOUSEEVENTF_LEFTUP
        elif hex_key_code == self.key['middle']:
            dwflags_down = self.__MOUSEEVENTF_MIDDLEDOWN
            dwflags_up = self.__MOUSEEVENTF_MIDDLEUP
        elif hex_key_code == self.key['right']:
            dwflags_down = self.__MOUSEEVENTF_RIGHTDOWN
            dwflags_up = self.__MOUSEEVENTF_RIGHTUP
        else:
            raise ValueError("Error: Mouse click value error!")

        return (dwflags_down, dwflags_up)

    def click(self, hex_key_code):
        dwflags_down, dwflags_up = self.__mapKey(hex_key_code)

        self.__operateMouse(dwflags_down)
        self.__operateMouse(dwflags_up)

    def doubleClick(self, hex_key_code):
        self.click(hex_key_code)
        time.sleep(self.time_click_interval)
        self.click(hex_key_code)

    def drag(self, hex_key_code, sx, sy, ex, ey):
        dwflags_down, dwflags_up = self.__mapKey(hex_key_code)
        self.move(sx, sy)
        self.__operateMouse(dwflags_down)
        self.move(ex, ey)
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
    action = Action()
    # action.keyboard.shortKey(action.keyboard.key['I'])
    tx, ty = action.mouse.getPosition()
    print(tx, ty)
    print(action.mouse.getPosition())
    print(action.mouse.getScreenSize())
    # action.mouse.moveReletiveFast(100, 200)
    # action.mouse.moveAbsoluteFast(100, 200)
    # action.mouse.moveReletive(100, 200)
    while True:
        time.sleep(0.5)
        action.keyboard.duringKey(action.keyboard.key['E'], 0.8)
        # print(action.mouse.getPosition())
        # action.mouse.move(1000, 500)
        # print(action.mouse.getPosition())
    # action.mouse.click(action.mouse.key['right'])
    # action.mouse.drag(action.mouse.key['left'], 500, 300, 800, 500)

    # print(action.mouse.generateRandlist(10, 1))
    # print(action.mouse.generateRandlist(10, 1))

    # while True:
    #     time.sleep(3)
    #     print(get_mpos())
    #     set_mpos((100, 200))
    #     mouse_press((100, 200))
    #     time.sleep(2)
    #     mouse_release((100, 200))
    # time.sleep(3)
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
