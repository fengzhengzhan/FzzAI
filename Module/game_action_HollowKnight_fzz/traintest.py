from dependencies import *


class ModelTrainTest:
    def __init__(self, path_autosave):
        # 初始化环境
        path_game_autosave = os.path.join(path_autosave, confhk.PATH_DATASET)
        # print(path_game_autosave)
        if not os.path.exists(path_game_autosave):
            os.makedirs(path_game_autosave)


    def train(self):
        for epoch in range(confhk.EPOCH):
            # 初始化每次场景启动的参数

            # 进入攻击场景
            while True:
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
    ModelTrainTest("C:\Files\Data\Life\Code\GithubCode\FzzAI\dataset_autosave")