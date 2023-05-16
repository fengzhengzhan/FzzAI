from dependencies import *


class CharacterStatus:
    def __init__(self, read_memory):
        """
        Parameters:
            read_memory: 需要ReadMemory object
        """
        self.blood = 9  # 人物血量
        self.power = 0  # 人物能量
        self.x = None
        self.y = None

        self.alive = True  # 人物是否还存活

        self.read_memory = read_memory

    ######################
    # 更新血量
    ######################
    def __gainBloodFromScene(self, grey_scene):
        # 判定自己的血量
        blood_num = 0
        range_set = False
        for self_bd_num in grey_scene[0]:
            if self_bd_num > 215 and range_set:
                blood_num += 1
                range_set = False
            elif self_bd_num < 55:
                range_set = True
        self.blood = blood_num

    def __gainBloodFromMemory(self):
        return self.read_memory.gainValueFromMuladdr(
            "UnityPlayer.dll", 0x019D7CF0, [0x10, 0x100, 0x28, 0x68, 0x218, 0x190])

    def updateBlood(self):
        """int 0-9"""
        __blood = self.__gainBloodFromMemory()
        self.blood = __blood

    ######################
    # 更新能量
    ######################
    def __gainPowerFromScene(self, grey_scene, power_window):
        # 获取自己的能量
        # 判定自己的能量作为攻击boss的血量
        power_num = 0
        for i in range(0, power_window[3] - power_window[1]):
            self_power_num = grey_scene[i][0]
            if self_power_num > 90:
                power_num += 1
        return power_num

    def __gainPowerFromMemory(self):
        return self.read_memory.gainValueFromMuladdr(
            "UnityPlayer.dll", 0x019D7CF0, [0x10, 0x20, 0x0, 0x38, 0x150, 0x1CC])

    def updatePower(self):
        """int 0-99"""
        __power = self.__gainPowerFromMemory()
        self.power = __power

    ######################
    # 更新X
    ######################
    def __gainXFromMemory(self):
        return self.read_memory.gainValueFromMuladdr(
            "UnityPlayer.dll", 0x01A1DDD8, [0xA90, 0x640, 0x228, 0x1A0, 0x78, 0x2C])

    def updataX(self):
        """float"""
        __x = self.__gainXFromMemory()
        self.x = __x

    ######################
    # 更新所有状态
    ######################
    def updateStatus(self):
        self.updateBlood()
        self.updatePower()
        self.updataX()


if __name__ == '__main__':
    from Environment.read_momery import ReadMemory

    # CharacterStatus(ReadMemory("Hollow Knight")).updatePower()
    CharacterStatus(ReadMemory("Hollow Knight")).updataX()
