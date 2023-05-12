from dependencies import *


class CharacterStatus:
    def __init__(self):
        self.blood = None  # 人物血量
        self.power = None  # 人物能量
        self.alive = True  # 人物是否还存活

    def UpdataBloodFromScene(self, grey_scene):
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

    def UpdataPowerFromScene(self, grey_scene, power_window):
        # 获取自己的能量
        # 判定自己的能量作为攻击boss的血量
        power_num = 0
        for i in range(0, power_window[3] - power_window[1]):
            self_power_num = grey_scene[i][0]
            if self_power_num > 90:
                power_num += 1
        self.power = power_num

    def UpdateStatus(self):
        pass
