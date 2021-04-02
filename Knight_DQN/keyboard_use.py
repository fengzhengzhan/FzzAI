import ctypes
import time
from handle_top import handld_top
import cv2
# from DQN_knight_train import self_blood_number
from argconfig import *
from grep_sreen import grab_screen

SendInput = ctypes.windll.user32.SendInput


# C struct defs

PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort),
    ]


class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]


# Actuals Functions
def self_blood_number(self_gray):
    self_blood = 0
    range_set = False
    # print(self_gray[0])
    # print(self_gray, self_gray.shape)
    for self_bd_num in self_gray[0]:
        # 大于215判定为一格血量,范围取值，血量面具为高亮
        # print(self_bd_num, end=" ")
        if self_bd_num > 215 and range_set:
            self_blood += 1
            range_set = False
        elif self_bd_num < 55:
            range_set = True
    # print(self_blood)
    return self_blood

def PressKey(hexKeyCode) -> None:
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def ReleaseKey(hexKeyCode) -> None:
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


# use keyboard
I = 0x17
J = 0x24
K = 0x25
L = 0x26

SPACE = 0x39

Q = 0x10
W = 0x11
E = 0x12
R = 0x13

ENTER = 0x1C
T = 0x14
ESC = 0x01

# 移动
def i_up():
    PressKey(I)
    time.sleep(0.1)
    ReleaseKey(I)
    # time.sleep(0.2)

# def j_left():
#     PressKey(J)
#     time.sleep(0.1)
#     ReleaseKey(J)
#     PressKey(R)
#     time.sleep(0.05)
#     ReleaseKey(R)
#     # time.sleep(0.2)

def j_left():
    PressKey(J)
    time.sleep(0.1)
    ReleaseKey(J)
    # PressKey(R)
    # time.sleep(0.05)
    # ReleaseKey(R)
    # time.sleep(0.2)

def k_down():
    PressKey(K)
    time.sleep(0.15)
    ReleaseKey(K)
    # time.sleep(0.2)

# def l_right():
#     PressKey(L)
#     time.sleep(0.1)
#     ReleaseKey(L)
#     PressKey(R)
#     time.sleep(0.05)
#     ReleaseKey(R)
#     # time.sleep(0.2)

def l_right():
    PressKey(L)
    time.sleep(0.1)
    ReleaseKey(L)
    # PressKey(R)
    # time.sleep(0.05)
    # ReleaseKey(R)
    # time.sleep(0.2)

def i_attack():
    PressKey(I)
    # time.sleep(0.05)
    PressKey(R)
    time.sleep(0.02)
    ReleaseKey(I)
    ReleaseKey(R)


def k_attack():
    PressKey(K)
    PressKey(R)
    time.sleep(0.02)
    ReleaseKey(K)
    ReleaseKey(R)

# 跳跃
def space_long_up():
    PressKey(SPACE)
    time.sleep(0.45)
    ReleaseKey(SPACE)
    # time.sleep(0.2)

def space_short_up():
    PressKey(SPACE)
    time.sleep(0.2)
    ReleaseKey(SPACE)
    # time.sleep(0.2)

def space_stay():
    PressKey(SPACE)
    time.sleep(0.02)


def to_left():
    PressKey(J)
    time.sleep(0.02)

def to_right():
    PressKey(L)
    time.sleep(0.02)

# 技能
def q_short_use():
    # TODO test
    PressKey(Q)
    time.sleep(0.02)
    ReleaseKey(Q)
    # time.sleep(0.2)

def q_long_use():
    PressKey(Q)
    time.sleep(1.5)
    ReleaseKey(Q)
    # time.sleep(0.2)

# 攻击
def r_short_attack():
    PressKey(R)
    time.sleep(0.02)
    ReleaseKey(R)
    # time.sleep(0.2)

def r_doubleshort_attack():
    PressKey(R)
    time.sleep(0.02)
    ReleaseKey(R)
    time.sleep(0.24)
    PressKey(R)
    time.sleep(0.02)
    ReleaseKey(R)
    # time.sleep(0.2)

# def r_long_attack():
#     PressKey(R)
#     time.sleep(1.6)
#     ReleaseKey(R)
#     # time.sleep(0.2)

# 运动
def w_long_run():
    # TODO test
    PressKey(W)
    time.sleep(0.5)
    ReleaseKey(W)
    # time.sleep(0.2)

def e_run():
    PressKey(E)
    time.sleep(0.02)
    ReleaseKey(E)
    # time.sleep(0.2)

# 技能
def iq_boom():
    PressKey(I)
    PressKey(Q)
    time.sleep(0.02)
    # time.sleep(0.05)
    ReleaseKey(Q)
    ReleaseKey(I)

def jq_boom():
    PressKey(J)
    time.sleep(0.02)
    PressKey(Q)
    # time.sleep(0.05)
    ReleaseKey(Q)
    ReleaseKey(J)

def kq_doom():
    PressKey(K)
    PressKey(Q)
    time.sleep(0.02)
    ReleaseKey(Q)
    ReleaseKey(K)


def lq_boom():
    PressKey(L)
    time.sleep(0.02)
    PressKey(Q)
    # time.sleep(0.05)
    ReleaseKey(Q)
    ReleaseKey(L)

def t_pause():
    PressKey(T)
    time.sleep(0.05)
    ReleaseKey(T)
    # time.sleep(0.2)

def esc_quit():
    PressKey(ESC)
    time.sleep(0.05)
    ReleaseKey(ESC)
    # time.sleep(0.2)

# def init_start():
#     print("[-] Knight dead,restart...")
#     # 椅子到通道
#     time.sleep(3)
#     PressKey(J)
#     time.sleep(0.05)
#     ReleaseKey(J)
#     time.sleep(2.5)
#     PressKey(W)
#     time.sleep(1.3)
#     ReleaseKey(W)
#     time.sleep(1.9)
#     PressKey(W)
#     time.sleep(0.05)
#     ReleaseKey(W)
#     time.sleep(2)
#
#     # 通道
#     PressKey(SPACE)
#     time.sleep(0.5)
#     ReleaseKey(SPACE)
#     time.sleep(0.05)
#
#     PressKey(SPACE)
#     PressKey(J)
#     time.sleep(0.5)
#     ReleaseKey(SPACE)
#
#     time.sleep(0.05)
#     PressKey(SPACE)
#     time.sleep(0.05)
#     ReleaseKey(SPACE)
#     time.sleep(0.05)
#
#     PressKey(SPACE)
#     time.sleep(0.05)
#     ReleaseKey(SPACE)
#     time.sleep(0.05)
#
#     # PressKey(SPACE)
#     # time.sleep(0.05)
#     # ReleaseKey(SPACE)
#     PressKey(SPACE)
#     time.sleep(0.5)
#     ReleaseKey(SPACE)
#     ReleaseKey(J)
#
#     # 左拿徽章
#     time.sleep(4.5)
#     PressKey(J)
#     time.sleep(0.4)
#     ReleaseKey(J)
#
#     # 通行证
#     PressKey(I)
#     time.sleep(0.05)
#     ReleaseKey(I)
#     time.sleep(1.5)
#     PressKey(J)
#     time.sleep(0.05)
#     ReleaseKey(J)
#     time.sleep(0.05)
#     PressKey(ENTER)
#     time.sleep(0.05)
#     ReleaseKey(ENTER)
#
#     # 右进角斗场
#     time.sleep(2.5)
#     PressKey(L)
#     time.sleep(0.05)
#     ReleaseKey(L)
#     time.sleep(0.05)
#     PressKey(W)
#     time.sleep(1.3)
#     ReleaseKey(W)
#     time.sleep(7.5)
#     PressKey(W)
#     time.sleep(0.05)
#     ReleaseKey(W)
#     print("[+] Enter start...")

def init_init():
    time.sleep(3)
    PressKey(I)
    time.sleep(0.05)
    ReleaseKey(I)
    time.sleep(1)
    PressKey(SPACE)
    time.sleep(0.05)
    ReleaseKey(SPACE)
    time.sleep(6)
    print("[+] Enter start...")


def init_start():
    print("[-] Knight dead,restart...")
    # # 椅子到格林
    # time.sleep(3)
    # PressKey(L)
    # time.sleep(0.05)
    # ReleaseKey(L)
    # time.sleep(1.5)
    # PressKey(W)
    # time.sleep(1.3)
    # ReleaseKey(W)
    # time.sleep(5.9)
    # PressKey(W)
    # time.sleep(0.05)
    # ReleaseKey(W)
    # time.sleep(2)

    while True:
        handld_top()
        blood_window_gray_double = cv2.cvtColor(grab_screen(blood_window), cv2.COLOR_RGBA2GRAY)
        self_blood_double = self_blood_number(blood_window_gray_double)
        if self_blood_double == 9:
            break
        time.sleep(1)

    time.sleep(2)
    # 起身
    handld_top()
    PressKey(SPACE)
    time.sleep(0.05)
    ReleaseKey(SPACE)
    time.sleep(2)

    handld_top()
    PressKey(SPACE)
    time.sleep(0.05)
    ReleaseKey(SPACE)
    time.sleep(2)


    # 选择
    handld_top()
    PressKey(I)
    time.sleep(0.05)
    ReleaseKey(I)
    time.sleep(1)
    handld_top()
    PressKey(SPACE)
    time.sleep(0.05)
    ReleaseKey(SPACE)
    time.sleep(5)
    print("[+] Enter start...")




# examples
if __name__ == "__main__":
    # time.sleep(3)
    # while True:
    time.sleep(1)
    iq_boom()
    time.sleep(1)
    kq_doom()
    # space_short_up()
    # ReleaseKey(SPACE)
    # ReleaseKey(J)
    # ReleaseKey(L)
    # ReleaseKey(R)
    # ReleaseKey(R)
    # ReleaseKey(R)
    # ReleaseKey(R)
    # r_doubleshort_attack()
    # time.sleep(0.05)
    # l_right()
    # l_right()
    # time.sleep(1)
    # init_start()
    # while True:
    #     r_short_attack()
    # space_short_up()
    # k_attack()
    # while True:  # 16种操作
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


