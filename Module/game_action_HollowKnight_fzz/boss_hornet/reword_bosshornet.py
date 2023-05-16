from dependencies import *

class Reward:
    def gainReward(self):
        pass
        # return_reward = 0
        # stop = 0
        # if next_self_blood < 1:  # 自己死亡
        #     if emergence_break < 2 and self.all_blood >= 3:
        #         reward = 1
        #         done = 1
        #         emergence_break = 0
        #         return init_time, reward, done, stop, emergence_break
        #     elif emergence_break < 2:
        #         reward = -0.8
        #         done = 1
        #         emergence_break = 0
        #         return init_time, reward, done, stop, emergence_break
        #     else:
        #         reward = -0.8
        #         done = 1
        #         emergence_break = 100
        #         return init_time, reward, done, stop, emergence_break
        #
        # done = 0
        # emergence_break = 0
        #
        # if self.all_blood - next_self_blood >= 1:  # 费血
        #     if emergence_break < 2:
        #         return_reward += -0.5 * (self.all_blood - next_self_blood)
        #         self.all_blood = next_self_blood
        #     else:
        #         return_reward += -0.5 * (self.all_blood - next_self_blood)
        #         self.all_blood = next_self_blood
        #         emergence_break = 100
        # elif next_self_blood - self.all_blood >= 1:  # 回血
        #     if emergence_break < 2:
        #         return_reward += 0.6 * (next_self_blood - self.all_blood)
        #         self.all_blood = next_self_blood
        #     else:
        #         return_reward += 0.6 * (next_self_blood - self.all_blood)
        #         self.all_blood = next_self_blood
        #         emergence_break = 100
        #
        # # 早期进行攻击也奖励，已取消
        # # if action == I_ATTACK or action == J_LEFT or action == K_ATTACK or action == L_RIGHT or action == ATTACK_NUM:  # 攻击给予正反馈
        # #     if emergence_break < 2:
        # #         return_reward += 8
        # #     else:
        # #         return_reward += 8
        # #         emergence_break = 100
        #
        # if next_self_power - self.all_power >= 5:  # 增加能量
        #     if emergence_break < 2:
        #         return_reward += 0.4
        #         self.all_power = next_self_power
        #     else:
        #         return_reward += 0.4
        #         self.all_power = next_self_power
        #         emergence_break = 100
        #
        # if self.all_power - next_self_power >= 4:
        #     self.all_power = next_self_power
        #
        # # 早期增加时间也给予奖励，已经取消
        # # if emergence_break < 2:
        # #     return_reward += time.time() - init_time
        # #     init_time = time.time()
        # # else:
        # #     return_reward += time.time() - init_time
        # #     init_time = time.time()
        # #     emergence_break = 100
        #
        # return init_time, return_reward, done, stop, emergence_break



if __name__ == '__main__':
    pass

