import ctypes
import time
from Twindow_handletop import handle_top
import cv2
from config import *
from torch.distributions import Categorical
import numpy as np


# C struct defs
SendInput = ctypes.windll.user32.SendInput
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
A = 0x1E
D = 0x20

SPACE = 0x39

P = 0x19
ENTER = 0x1C
ESC = 0x01

# 抽样选择动作
def sample_select_action(x):
    # Vector
    # 处理一维数组
    x_max = np.max(x)
    x -= x_max
    numerator = np.exp(x)
    denominator = 1.0 / np.sum(numerator)
    x = numerator.dot(denominator)

    x = torch.as_tensor(x)  # 生成分布
    x_m = Categorical(x)  # 从分布中采样
    x_a = x_m.sample()

    return x_a.item()


# 箭头移动  0.4s半圈
def move(value, printflag):
    if value >= 0.0:
        PressKey(A)
        time.sleep(value)
        ReleaseKey(A)
        if printflag:
            print(" 操作向左->{}s".format(value))
    elif value < 0.0:
        PressKey(D)
        time.sleep(-value)
        ReleaseKey(D)
        if printflag:
            print(" 操作向右->{}s".format(-value))

# 辅助功能函数
def p_pause():
    PressKey(P)
    time.sleep(0.05)
    ReleaseKey(P)

def esc_quit():
    PressKey(ESC)
    time.sleep(0.05)
    ReleaseKey(ESC)


def init_startgame():
    PressKey(SPACE)
    time.sleep(0.05)
    ReleaseKey(SPACE)
    time.sleep(0.05)
    print("[+] Space start...")


# examples
if __name__ == "__main__":
    handle_top()
    init_startgame()
    move(value=0.4, printflag=True)

