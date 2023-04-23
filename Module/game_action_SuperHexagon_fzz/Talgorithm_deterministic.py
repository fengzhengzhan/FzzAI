from concurrent.futures import ThreadPoolExecutor

from Abody_keyboard import *
from Abrain_model import *
from Aeye_grabscreen import *
from Amemory_replaybuffer import *
from config import *
from Escore_readram import *
from Tlogo_print import *
from Twindow_handletop import *


if __name__ == '__main__':  # 多进程freeze_support()

    # print_logo()
    handle_top()
    screen = AeyeGrabscreen()  # 视觉模块：屏幕截图  先将视觉进程打开，同步截取屏幕
    score = EscoreReadram()  # 获取得分

    pro_step_num = 1

    for episode in range(1, EPISODES):
        handle_top()
        game_score = 0
        pro_game_score = 0
        reward = 0
        all_reward = 0
        step_num = 1
        pro_action_num = -1
        global_best_reward = GLOBAL_BEST_REWARD
        ti = time.strftime("%Y%m%d%H%M%S", time.localtime())  # 用于保存每轮的唯一标识，以便区分reward
        reward_flag = int(ti) + int(episode*1e14)
        print()
        print(reward_flag)

        init_startgame()
        while True:
            time.sleep(0.1)
            # Step1: 首次抓取屏幕
            _, area = screen.getstate()  # (N,C,H,W)
            # 进入下一状态
            # print(area.shape, area.device)  # torch.Size([1, 1, 238, 384]) cuda:0

            # Step2: 执行动作
            action = 3
            area = np.array(area.to(CPUDEVICE))[0]
            if 0.9 <= area[action] <= 1.1:
                if 0.1 <= area[action - 1] <= 0.3:
                    action = action - 1
                elif 0.1 <= area[action + 1] <= 0.3:
                    action = action + 1
                elif 0.1 <= area[action - 2] <= 0.3:
                    action = action - 2
                elif 0.1 <= area[action + 2] <= 0.3:
                    action = action + 2
                elif 0.1 <= area[action - 3] <= 0.3:
                    action = action - 3
            move_action = ACTION_STEPS[action]
            move(move_action, True)
            print(area)
            game_score = score.gainValue()
            # all_reward += reward

            pro_action_num = action
            step_num += 1

            # Step3: 动作完成，抓取屏幕，完成一次交d互 将最后一步默认不添加至进程池
            # 线程池->经验池存储
            # _, next_area = screen.getstate()
            if game_score - pro_game_score > END_GAME_TIME:
                pro_game_score = game_score
            else:
                # 游戏结束
                # print(game_score, pro_game_score)
                pro_step_num = step_num
                break


        # 训练模型，保存数据
        print("[*] 游戏轮数:", episode, "此轮奖励(持续时间):", reward)
        time.sleep(SLEEP_INTERVAL)

