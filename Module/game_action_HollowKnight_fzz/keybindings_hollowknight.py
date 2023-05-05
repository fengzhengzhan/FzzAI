from dependencies import *

# Actuals Functions
class KeyBingdings:
    def __init__(self):
        pass



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


if __name__ == '__main__':
    pass
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
    time.sleep(0.15)
    ReleaseKey(I)
    # time.sleep(0.2)

def j_left():
    PressKey(J)
    time.sleep(0.1)
    ReleaseKey(J)
    PressKey(R)
    time.sleep(0.05)
    ReleaseKey(R)
    # time.sleep(0.2)

# def j_left():
#     PressKey(J)
#     time.sleep(0.1)
#     ReleaseKey(J)
#     # PressKey(R)
#     # time.sleep(0.05)
#     # ReleaseKey(R)
#     # time.sleep(0.2)

def k_down():
    PressKey(K)
    time.sleep(0.15)
    ReleaseKey(K)
    # time.sleep(0.2)

def l_right():
    PressKey(L)
    time.sleep(0.1)
    ReleaseKey(L)
    PressKey(R)
    time.sleep(0.05)
    ReleaseKey(R)
    # time.sleep(0.2)

# def l_right():
#     PressKey(L)
#     time.sleep(0.1)
#     ReleaseKey(L)
#     # PressKey(R)
#     # time.sleep(0.05)
#     # ReleaseKey(R)
#     # time.sleep(0.2)

def i_attack():
    PressKey(I)
    # time.sleep(0.05)
    PressKey(R)
    time.sleep(0.05)
    ReleaseKey(I)
    ReleaseKey(R)


def k_attack():
    PressKey(K)
    # time.sleep(0.05)
    PressKey(R)
    time.sleep(0.05)
    ReleaseKey(K)
    ReleaseKey(R)

# 跳跃
def space_long_up():
    PressKey(SPACE)
    time.sleep(0.5)
    ReleaseKey(SPACE)
    # time.sleep(0.2)

def space_short_up():
    PressKey(SPACE)
    time.sleep(0.15)
    ReleaseKey(SPACE)
    # time.sleep(0.2)

# 技能
def q_short_use():
    # TODO test
    PressKey(Q)
    time.sleep(0.05)
    ReleaseKey(Q)
    # time.sleep(0.2)

def q_long_use():
    PressKey(Q)
    time.sleep(1.4)
    ReleaseKey(Q)
    # time.sleep(0.2)

# 攻击
def r_short_attack():
    PressKey(R)
    time.sleep(0.05)
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
    time.sleep(0.05)
    ReleaseKey(E)
    # time.sleep(0.2)

# 技能
def iq_boom():
    PressKey(I)
    PressKey(Q)
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

# 竞技场
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

# 神居
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

# 神居
def init_start():
    print("[-] Knight dead,restart...")
    # 起身
    PressKey(SPACE)
    time.sleep(0.05)
    ReleaseKey(SPACE)
    time.sleep(2)

    PressKey(SPACE)
    time.sleep(0.05)
    ReleaseKey(SPACE)
    time.sleep(2)


    # 选择
    PressKey(I)
    time.sleep(0.05)
    ReleaseKey(I)
    time.sleep(1)
    PressKey(SPACE)
    time.sleep(0.05)
    ReleaseKey(SPACE)
    time.sleep(6)
    print("[+] Enter start...")



# examples
if __name__ == "__main__":
    time.sleep(3)
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


def take_action(action):
    # 所有的攻击列表 10种攻击
    # ['i', 'j', 'k', 'l', 'r', 'ss', 'sl', 'qs', 'ql', 'e']
    if action == I_ATTACK:
        i_attack()
    elif action == J_LEFT:
        j_left()
    elif action == K_ATTACK:
        k_attack()
    elif action == L_RIGHT:
        l_right()
    elif action == ATTACK_NUM:
        r_short_attack()
    elif action == 5:
        space_short_up()
    elif action == 6:
        space_long_up()
    elif action == 7:
        q_short_use()
    elif action == 8:
        q_long_use()
    elif action == 9:
        e_run()