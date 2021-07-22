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
    torch.cuda.empty_cache()

    # print_logo()
    handle_top()
    screen = AeyeGrabscreen()  # 视觉模块：屏幕截图  先将视觉进程打开，同步截取屏幕
    agent = AbrainModel()  # 模型
    agent.load()
    replay = AmemoryReplaybuffer()  # 经验池回放
    replay.load()
    score = EscoreReadram()  # 获取得分
    pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)  # 线程池

    pro_step_num = 1

    for episode in range(1, EPISODES):
        handle_top()
        game_score = 0
        pro_game_score = 0
        reward = 0
        step_num = 1
        pro_action_num = -1
        global_best_reward = GLOBAL_BEST_REWARD
        ti = time.strftime("%Y%m%d%H%M%S", time.localtime())  # 用于保存每轮的唯一标识，以便区分reward
        reward_flag = int(ti) + int(episode*1e14)
        print()
        print(reward_flag)

        init_startgame()
        while True:
            # Step1: 首次抓取屏幕
            state = screen.getstate()  # (N,C,H,W)
            # 进入下一状态
            # print(state.shape, state.device)  # torch.Size([1, 1, 238, 384]) cuda:0

            # Step2: 执行动作
            action = agent.choose_action(state, pro_step_num, step_num)
            # print(action)
            move_action = ACTION_STEPS[action]
            move(move_action, True)
            game_score = score.get_score()
            reward = agent.action_judge(game_score, pro_action_num, action)

            pro_action_num = action
            step_num += 1


            # Step3: 动作完成，抓取屏幕，完成一次交d互 将最后一步默认不添加至进程池
            # 线程池->经验池存储
            next_state = screen.getstate()
            if game_score - pro_game_score > END_GAME_TIME:
                pool.submit(replay.Storage_thread, state, action, reward_flag, next_state, True, reward)
                pro_game_score = game_score
            else:
                # 游戏结束
                # print(game_score, pro_game_score)
                lastreward = -round(float((int(game_score) / 60) * 0.1), 2)
                pool.submit(replay.Storage_thread, state, action, reward_flag, next_state, True, lastreward)
                replay.push_reward(reward_flag=reward_flag, reward=reward)
                pro_step_num = step_num
                break


        # 训练模型，保存数据
        storagelen, posstoragelen, rewardflag_dictlen = replay.length()
        print("[*] 游戏轮数:", episode, "此轮奖励(持续时间):", reward)

        # 训练模型，否则应该休眠时间等待游戏准备
        if storagelen >= TRAINSTORAGELEN and episode >= TRAINEPISODELEN and not TEST_MODE:
            print("[-] 正在训练，持续时间", TRAININGDURATION ,"s...")
            starttime = time.time()
            startnum = 0
            while time.time() - starttime <= TRAININGDURATION and startnum < TRAINPERNUM:
                startnum += 1
                agent.train_network(replay=replay, lossprintflag=True, num_step=episode, per_step=startnum)
        else:
            time.sleep(SLEEP_GAME)

        time.sleep(SLEEP_INTERVAL)
        # 保存数据
        if episode == SAVE_FIRST_EPISODE or episode % SAVE_EPISODE == 0:
            agent.save()
            replay.save()
        if reward > global_best_reward:
            global_best_reward = reward
            agent.save_excellent(reward_flag)

    pool.shutdown()






