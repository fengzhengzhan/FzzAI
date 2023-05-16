from dependencies import *


class BossStatus:
    def __init__(self, read_memory):
        """
        Parameters:
            read_memory: 需要ReadMemory object
        """
        self.blood = 900  # 人物血量
        self.x = None
        self.y = None

        self.alive = True  # 人物是否还存活

        self.read_memory = read_memory

    ######################
    # 更新血量
    ######################
    def __updataBloodFromMemory(self):
        return self.read_memory.gainValueFromMuladdr(
            "UnityPlayer.dll", 0x019D4478, [0xD70, 0x238, 0x130, 0x28, 0x140])

    def updateBlood(self):
        """int 0-900"""
        __blood = self.__updataBloodFromMemory()
        self.blood = __blood

    ######################
    # 更新所有状态
    ######################
    def updateStatus(self):
        self.updateBlood()


if __name__ == '__main__':
    from Environment.read_momery import ReadMemory

    BossStatus(ReadMemory("Hollow Knight")).updateBlood()
