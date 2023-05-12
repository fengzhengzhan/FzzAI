from dependencies import *

from Agent.agent import Agent


class PrepareBindings:
    def __init__(self):
        self.__action = Agent().inputAction(has_keyboard=True, has_mouse=False)

    # 竞技场：愚人斗兽场
    def scenarioColosseumOfFools(self):
        projlog(INFO, "[-] Knight dead, restart in the ColosseumOfFools...")
        # 椅子到通道
        time.sleep(3)
        self.__action.keyboard.duringKey('J', 0.05)
        time.sleep(2.5)
        self.__action.keyboard.duringKey('W', 1.3)
        time.sleep(1.9)
        self.__action.keyboard.duringKey('W', 0.05)
        time.sleep(2)

        # 通道
        self.__action.keyboard.duringKey('Space', 0.5)
        time.sleep(0.05)

        self.__action.keyboard.pressKey('Space')
        self.__action.keyboard.pressKey('J')
        time.sleep(0.5)
        self.__action.keyboard.releaseKey('Space')

        time.sleep(0.05)
        self.__action.keyboard.duringKey('Space', 0.05)
        time.sleep(0.05)

        self.__action.keyboard.duringKey('Space', 0.05)
        time.sleep(0.05)

        # PressKey(SPACE)
        # time.sleep(0.05)
        # ReleaseKey(SPACE)
        self.__action.keyboard.duringKey('Space', 0.5)
        self.__action.keyboard.releaseKey('J')

        # 左拿徽章
        time.sleep(4.5)
        self.__action.keyboard.duringKey('J', 0.4)

        # 通行证
        self.__action.keyboard.duringKey('I', 0.05)
        time.sleep(1.5)
        self.__action.keyboard.duringKey('J', 0.05)
        time.sleep(0.05)
        self.__action.keyboard.duringKey('Enter', 0.05)

        # 右进角斗场
        time.sleep(2.5)
        self.__action.keyboard.duringKey('L', 0.05)
        time.sleep(0.05)
        self.__action.keyboard.duringKey('W', 1.3)
        time.sleep(7.5)
        self.__action.keyboard.duringKey('W', 0.05)
        projlog(INFO, "[+] Finish start in the ColosseumOfFools...")

    # 神居
    def scenarioPantheonsInit(self):
        time.sleep(3)
        self.__action.keyboard.duringKey('I', 0.05)
        time.sleep(1)
        self.__action.keyboard.duringKey('Space', 0.05)
        time.sleep(6)
        projlog(INFO, "[+] Finish init in the Pantheons...")

    # 神居
    def scenarioPantheonsRestart(self):
        projlog(INFO, "[-] Knight dead, restart in the Pantheons...")
        # 起身
        self.__action.keyboard.duringKey('Space', 0.05)
        time.sleep(2)

        self.__action.keyboard.duringKey('Space', 0.05)
        time.sleep(2)

        # 选择
        self.__action.keyboard.duringKey('I', 0.05)
        time.sleep(1)
        self.__action.keyboard.duringKey('Space', 0.05)
        time.sleep(6)
        projlog(INFO, "[+] Finish start in the Pantheons...")


class OperationBindings:
    def __init__(self):
        self.__action = Agent().inputAction(has_keyboard=True, has_mouse=False)

    # 移动
    def iUp(self):
        self.__action.keyboard.duringKey('I', 0.15)
        # time.sleep(0.2)

    def jLeft(self):
        self.__action.keyboard.duringKey('J', 0.1)
        self.__action.keyboard.duringKey('R', 0.05)

    def kDown(self):
        self.__action.keyboard.duringKey('K', 0.15)

    def lRight(self):
        self.__action.keyboard.duringKey('L', 0.1)
        self.__action.keyboard.duringKey('R', 0.05)

    # 跳跃
    def spaceShort(self):
        self.__action.keyboard.duringKey('Space', 0.15)

    def spaceLong(self):
        self.__action.keyboard.duringKey('Space', 0.5)

    # 运动
    def wLongRun(self):
        self.__action.keyboard.duringKey('W', 0.5)

    def eRun(self):
        self.__action.keyboard.duringKey('E', 0.05)

    # 攻击
    def iAttack(self):
        self.__action.keyboard.pressKey('I')
        # time.sleep(0.05)
        self.__action.keyboard.pressKey('R')
        time.sleep(0.05)
        self.__action.keyboard.releaseKey('I')
        self.__action.keyboard.releaseKey('R')

    def kAttack(self):
        self.__action.keyboard.pressKey('K')
        # time.sleep(0.05)
        self.__action.keyboard.pressKey('R')
        time.sleep(0.05)
        self.__action.keyboard.releaseKey('K')
        self.__action.keyboard.releaseKey('R')

    def rShortAttack(self):
        self.__action.keyboard.duringKey('R', 0.05)

    def rLongAttack(self):
        self.__action.keyboard.duringKey('R', 1.6)

    # 技能
    def qShortUse(self):
        self.__action.keyboard.duringKey('Q', 0.05)

    def qLongUse(self):
        self.__action.keyboard.duringKey('Q', 1.4)

    def iqBoom(self):
        self.__action.keyboard.pressKey('I')
        self.__action.keyboard.pressKey('Q')
        time.sleep(0.05)
        self.__action.keyboard.releaseKey('I')
        self.__action.keyboard.releaseKey('Q')

    def jqBoom(self):
        self.__action.keyboard.pressKey('J')
        time.sleep(0.02)
        self.__action.keyboard.pressKey('Q')
        # time.sleep(0.05)
        self.__action.keyboard.releaseKey('J')
        self.__action.keyboard.releaseKey('Q')

    def kqBoom(self):
        self.__action.keyboard.pressKey('K')
        self.__action.keyboard.pressKey('Q')
        time.sleep(0.05)
        self.__action.keyboard.releaseKey('K')
        self.__action.keyboard.releaseKey('Q')

    def lqBoom(self):
        self.__action.keyboard.pressKey('L')
        time.sleep(0.02)
        self.__action.keyboard.pressKey('Q')
        # time.sleep(0.05)
        self.__action.keyboard.releaseKey('L')
        self.__action.keyboard.releaseKey('Q')

    def tPause(self):
        self.__action.keyboard.duringKey('T', 0.05)

    def escQuit(self):
        self.__action.keyboard.duringKey('ESC', 0.05)

    def ctrlsSave(self):
        self.__action.keyboard.pressKey('LeftCtrl')
        self.__action.keyboard.pressKey('S')
        time.sleep(0.05)
        self.__action.keyboard.releaseKey('LeftCtrl')
        self.__action.keyboard.releaseKey('S')


# Actuals Functions
class KeyBindings:
    def __init__(self):
        self.prepare_key = PrepareBindings()
        self.operation_key = OperationBindings()

        # 所有的攻击列表 10种攻击
        # ['i', 'j', 'k', 'l', 'r', 'ss', 'sl', 'qs', 'ql', 'e']
        # 注册函数，使用时调用，kb.option_10['i']()
        self.option_10 = {
            'i': self.operation_key.iAttack,
            'j': self.operation_key.jLeft,
            'k': self.operation_key.kAttack,
            'l': self.operation_key.lRight,
            'r': self.operation_key.rShortAttack,
            'ss': self.operation_key.spaceShort,
            'sl': self.operation_key.spaceLong,
            'qs': self.operation_key.qShortUse,
            'ql': self.operation_key.qLongUse,
            'e': self.operation_key.eRun,
        }


# examples
if __name__ == "__main__":
    time.sleep(3)
    kb = KeyBindings()
    # kb.option_10['i']()
    kb.prepare_key.scenarioPantheonsInit()  # i

    # while True:  # 16种操作全部测试
    #     kb.operation_key.iAttack()
    #     kb.operation_key.jLeft()
    #     time.sleep(1)
    #     kb.operation_key.kDown()
    #     kb.operation_key.lRight()
    #     kb.operation_key.spaceLong()
    #     kb.operation_key.spaceShort()
    #     kb.operation_key.qShortUse()
    #     kb.operation_key.qLongUse()
    #     kb.operation_key.rShortAttack()
    #     kb.operation_key.rLongAttack()
    #     kb.operation_key.wLongRun()
    #     kb.operation_key.eRun()
    #     kb.operation_key.iqBoom()
    #     kb.operation_key.kqBoom()
    #     kb.operation_key.jqBoom()
    #     kb.operation_key.lqBoom()
    #     time.sleep(1)
