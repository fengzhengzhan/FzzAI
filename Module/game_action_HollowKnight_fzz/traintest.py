import time

from dependencies import *


class ModelTrainTest:
    def __init__(self, path_autosave):
        # 初始化环境
        path_game_autosave = os.path.join(path_autosave, confhk.PATH_DATASET)
        # Logger.log(DEBUG, path_game_autosave)
        if not os.path.exists(path_game_autosave):
            os.makedirs(path_game_autosave)

    def train(self):
        status = GlobalStatus()

        # 置顶程序并启动第一次的进入操作
        change_env = ChangeEnv()
        prepare_bindings = PrepareBindings()
        # Logger.log(DEBUG, reset.travelProcess())
        change_env.toppingProcess('无界面测试窗口.txt - 记事本', -10, 0, 1280, 720)

        # 进入战斗场景的前置操作（恢复环境）
        prepare_bindings.scenarioPantheonsInit()

        for epoch in range(confhk.EPOCH):
            # 初始化每次场景启动的参数

            # 初始化展示状态

            # 记录时间
            status.start_time = time.time()
            status.last_time = time.time()

            # 进入攻击场景
            while status.goon:
                print("test")
                # Step1: 获取环境

                # Step2: 执行动作
                # 建议每次相应时间<0.25即一秒钟4个操作，否则会感觉卡卡的，耗时按键模拟除外
                # Step3: 再次获取环境
                # Step4: 获得奖励
                # Step5: 保存数据和模型
                pass
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
    model_train_test = ModelTrainTest(ProjectPath.dateset_autosave_path)
    model_train_test.train()