from dependencies import *


## Structure
class StructKeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
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


class ActionKeyBoard:
    """
    键盘模拟
    KeyBoard Simulate
    """

    def __init__(self, short_during_time=0.05, long_during_time=0.2):
        self.__dwflags_presskey = 0x0008
        self.__dwflags_releasekey = 0x0008 | 0x0002

        self.__short_during_time = short_during_time
        self.__long_during_time = long_during_time

        self.key = {'I': 0x17,
                    'J': 0x24,
                    'K': 0x25,
                    'L': 0x26, }

    def __operateKey(self, hex_key_code, dwflags):
        uki = UnionKeyInput()
        uki.ki = StructKeyBdInput(0, hex_key_code, dwflags, 0, ctypes.pointer(ctypes.c_ulong(0)))
        si = StructInput(ctypes.c_ulong(1), uki)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(si), ctypes.sizeof(si))

    def __finKey(self, hex_key_code, during_time):
        self.__operateKey(hex_key_code, self.__dwflags_presskey)
        time.sleep(during_time)
        self.__operateKey(hex_key_code, self.__dwflags_releasekey)

    def shortKey(self, hex_key_code):
        self.__finKey(hex_key_code, self.__short_during_time)

    def longKey(self, hex_key_code):
        self.__finKey(hex_key_code, self.__long_during_time)

    def customTimeKey(self, hex_key_code, during_time):
        self.__finKey(hex_key_code, during_time)

    def doubleKey(self, hex_key_code):
        self.shortKey(hex_key_code)
        time.sleep(self.__short_during_time)
        self.shortKey(hex_key_code)


class ActionMouse:
    """
    鼠标模拟
    Mouse Simulate
    """
    pass


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


# SendInput = ctypes.windll.user32.SendInput
#
# PUL = ctypes.POINTER(ctypes.c_ulong)

#
# class KeyBdInput(ctypes.Structure):
#     _fields_ = [("wVk", ctypes.c_ushort),
#                 ("wScan", ctypes.c_ushort),
#                 ("dwFlags", ctypes.c_ulong),
#                 ("time", ctypes.c_ulong),
#                 ("dwExtraInfo", PUL)]
#
#
# class HardwareInput(ctypes.Structure):
#     _fields_ = [("uMsg", ctypes.c_ulong),
#                 ("wParamL", ctypes.c_short),
#                 ("wParamH", ctypes.c_ushort)]
#
#
# class MouseInput(ctypes.Structure):
#     _fields_ = [("dx", ctypes.c_long),
#                 ("dy", ctypes.c_long),
#                 ("mouseData", ctypes.c_ulong),
#                 ("dwFlags", ctypes.c_ulong),
#                 ("time", ctypes.c_ulong),
#                 ("dwExtraInfo", PUL)]
#
#
# class Input_I(ctypes.Union):
#     _fields_ = [("ki", KeyBdInput),
#                 ("mi", MouseInput),
#                 ("hi", HardwareInput)]
#
#
# class Input(ctypes.Structure):
#     _fields_ = [("type", ctypes.c_ulong),
#                 ("ii", Input_I)]


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_ulong), ("y", ctypes.c_ulong)]


# Actuals Functions

# def PressKey(hexKeyCode):
#     extra = ctypes.c_ulong(0)
#     FInputs = Input * 1
#     ii_ = Input_I()
#     ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
#     x = Input(ctypes.c_ulong(1), ii_)
#     ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
#
#
# def ReleaseKey(hexKeyCode):
#     extra = ctypes.c_ulong(0)
#     ii_ = Input_I()
#     ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
#     x = Input(ctypes.c_ulong(1), ii_)
#     ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def get_mpos():
    orig = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(orig))
    return int(orig.x), int(orig.y)


def set_mpos(pos):
    x, y = pos
    ctypes.windll.user32.SetCursorPos(x, y)


def mouse_press(pos, button='left'):
    i = 2
    if button == 'right':
        i = 8
    set_mpos(pos)
    extra = ctypes.c_ulong(0)
    ii_ = UnionKeyInput()
    ii_.mi = StructMouseInput(0, 0, 0, i, 0, ctypes.pointer(extra))
    x = StructInput(ctypes.c_ulong(0), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


# def mouse_release(pos, windowName=se, button='left'):
#     i = 4
#     if button == 'right':
#         i = 16
#     set_mpos(x, y)
#     extra = ctypes.c_ulong(0)
#     ii_ = Input_I()
#     ii_.mi = MouseInput(0, 0, 0, i, 0, ctypes.pointer(extra))
#     x = Input(ctypes.c_ulong(0), ii_)
#     ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def mouse_release(pos, button='left'):
    i = 4
    if button == 'right':
        i = 16
    set_mpos(pos)
    extra = ctypes.c_ulong(0)
    ii_ = UnionKeyInput()
    ii_.mi = StructMouseInput(0, 0, 0, i, 0, ctypes.pointer(extra))
    x = StructInput(ctypes.c_ulong(0), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


class Action:
    def __init__(self):
        self.keyboard = ActionKeyBoard()


if __name__ == "__main__":
    action = Action()
    action.keyboard.shortKey(action.keyboard.key['I'])
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
