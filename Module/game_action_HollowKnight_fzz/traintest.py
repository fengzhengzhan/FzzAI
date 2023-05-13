import time

from dependencies import *

from Agent.agent import Agent
from Environment.environment import Environment
from Tool.status_window import GlobalStatus

from agent_status import CharacterStatus
from boss_hornet.boss_status import BossStatus
from keybindings_hollowknight import PrepareBindings, OperationBindings

from boss_hornet.network_CNN import NetworkCNN
from boss_hornet.algorithm_DQN import AlgorithmDQN
from boss_hornet.reword_bosshornet import Reward


class ModelTrainTest:
    def __init__(self, path_autosave):
        # 初始化环境
        # 创建HollowKnight游戏数据目录
        path_game_autosave = os.path.join(path_autosave, confhk.PATH_DATASET)
        projlog(DEBUG, path_game_autosave)
        if not os.path.exists(path_game_autosave):
            os.makedirs(path_game_autosave)
            projlog(INFO, "[+] " + str(path_game_autosave))

    def train(self):
        # 初始化信息获取类
        agent = Agent()
        env = Environment()

        # 初始化状态类
        status = GlobalStatus()
        agent_status = CharacterStatus()
        boos_status = BossStatus()

        # 初始化个性化定义类
        change_env = env.outputChangeEnv()
        manager_scene = env.outputSceneManager()
        prepare_bindings = PrepareBindings()
        operation_bindings = OperationBindings()

        # 初始化网络模型
        policy_network = NetworkCNN(confhk.WIDTH, confhk.HEIGHT, confhk.ACTION_SIZE).to(confglobal.DEVICE)
        target_network = NetworkCNN(confhk.WIDTH, confhk.HEIGHT, confhk.ACTION_SIZE).to(confglobal.DEVICE)
        algorithm_dqn = AlgorithmDQN(policy_network, target_network)
        obj_reward = Reward()

        # 置顶程序并启动第一次的进入操作
        # projlog(DEBUG, change_env.travelProcess())
        # change_env.toppingProcess('无界面测试窗口.txt - 记事本', -10, 0, 1280, 720)
        change_env.toppingProcess('Hollow Knight', 0 - 8, 0, 1280 + 16, 720 + 39)

        # 进入战斗场景的前置操作（恢复环境）
        # prepare_bindings.scenarioPantheonsInit()
        # operation_bindings.ctrlsSave()

        for epoch in range(confhk.EPOCH):
            # 初始化每次场景启动的参数
            status.resetStatus()

            # 初始化展示状态
            # agent_status

            # 记录时间
            status.start_time = time.time()
            status.last_time = status.start_time

            # 进入攻击场景
            while status.goon:
                # Step1: 获取环境
                state = manager_scene.gainTransData()[0]  # cv2.COLOR_BGRA2BGR

                # Step2: 执行动作
                # 建议每次相应时间<0.25即一秒钟4个操作，否则会感觉卡卡的，耗时按键模拟除外
                algorithm_dqn.chooseAction(state, 0.5)

                # Step3: 再次获取环境
                next_state = manager_scene.gainTransData()[0]  # cv2.COLOR_BGRA2BGR
                # Step4: 获得奖励
                reward = obj_reward.gainReward()
                # Step5: 保存数据和模型

                # 　紧急暂停
                # def pause_game(paused):
                #     if paused:
                #         print("[-] paused")
                #         while True:
                #             keys = key_check()
                #             if 'T' in keys:
                #                 if paused:
                #                     paused = False
                #                     print("[+] start game")
                #                     esc_quit()
                #                     time.sleep(1)
                #                     break
                #                 else:
                #                     paused = True
                #                     esc_quit()
                #                     time.sleep(1)  # jw
                #     keys = key_check()
                #     if 'T' in keys:
                #         if paused:
                #             paused = False
                #             print("[+] start game")
                #             esc_quit()
                #             time.sleep(1)
                #         else:
                #             paused = True
                #             print("[-] pause game")
                #             esc_quit()
                #             time.sleep(1)
                #
                #     return paused

    def trainReplay(self):
        pass

    def test(self):
        pass


if __name__ == '__main__':
    model_train_test = ModelTrainTest(projectpath.dateset_autosave_path)
    model_train_test.train()
