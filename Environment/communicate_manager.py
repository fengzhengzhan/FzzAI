from dependencies import *

class CommunicateManager:
    """
    在多进程中使用manager传递信息
    Use Manager communicate information.
    """
    def __init__(self):
        # 分模式获得对应的视觉信息
        print(vision_list)
        if len(vision_list) == 1:
            # 当只获得一个视觉信息时，可以传入句柄获得额外的优化
            # 进程间共享变量
            manager = Manager()
            self.screen_index = manager.Value('i', MANAGER_LIST_LENGTH)  # 为防止数组越界
            self.screen_list = manager.list([0 for i in range(MANAGER_LIST_LENGTH)])
            self.area_list = manager.list([0 for i in range(MANAGER_LIST_LENGTH)])
        elif len(vision_list) > 1:
            pass


