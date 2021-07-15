import gym
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

from Abody_keyboard import *
from Abrain_model import *
from Aeye_grabscreen import *
from Amemory_replaybuffer import *
from Escore_readram import *
from Tlogo_print import *
from Twindow_handletop import *
from config import *


# 手动调整窗口位置
if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    torch.cuda.empty_cache()
    screen = AeyeGrabscreen()  # 视觉模块：屏幕截图  先将视觉进程打开，同步截取屏幕
    agent = AbrainModelDDPG()  # DDPG模型
    agent.load()
    replay = AmemoryReplaybuffer()  # 经验池回放
    replay.load()
    score = EscoreReadram()  # 获取得分
    pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)  # 线程池
    # time.sleep(2)

    for episode in range(1, EPISODES):
        s = env.reset()
        ep_reward = 0
        if episode == 1:
            for i in range(600):
                # print(i)
                if RENDER:
                    env.render()

        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()
            # Step1: 首次抓取屏幕
            state = screen.getstate().unsqueeze(0)  # (N,C,D,H,W)
            # print(state.shape, state.device)  # torch.Size([1, 1, 4, 238, 384]) cuda:0

            # Step2: 执行动作
            action = agent.choose_action(state, episode)
            action = np.array([action])
            observation, reward, done, info = env.step(action)
            # print(reward, done, info)
            ep_reward += reward

            # Step3: 动作完成，抓取屏幕，完成一次交互 将最后一步默认不添加至进程池
            # 线程池->经验池存储
            pool.submit(replay.Storage_thread, state, action, None, screen, True, reward)


        # 训练模型，保存数据
        storagelen, posstoragelen, rewardflag_dictlen = replay.length()
        print("[*] 游戏轮数:", episode, "此轮奖励(持续时间):", ep_reward)

        # 训练模型，否则应该休眠时间等待游戏准备
        if storagelen >= TRAINSTORAGELEN and episode >= TRAINEPISODELEN and not TEST_MODE:
            print("[-] 正在训练，持续时间", TRAININGDURATION ,"s...")
            starttime = time.time()
            while time.time() - starttime <= TRAININGDURATION:
                agent.train_network(replay=replay, lossprintflag=True, num_step=episode)
        else:
            time.sleep(SLEEP_GAME)

        # 保存数据
        if episode == SAVE_FIRST_EPISODE or episode % SAVE_EPISODE == 0:
            agent.save()
            replay.save()

    pool.shutdown()