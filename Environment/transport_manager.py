import time

from dependencies import *


class ProcessScreen(Process):
    """
    在多进程中使用manager传递信息
    Use Manager communicate information.
    """

    def __init__(self, channel_index, channel_data, channel_length):
        super(ProcessScreen, self).__init__()
        self.channel_index = channel_index
        self.channel_data = channel_data
        self.channel_length = channel_length
        self.process_switch = True

    def run(self):
        rs = ReadScreen()
        while True:
            # 多进程获取屏幕截图
            data = rs.converScreen()
            # 滚动Manager数组
            self.channel_data[(self.channel_index.value + 1) % self.channel_length] = data
            self.channel_index.value += 1


class TransportManager:
    def __init__(self, channel_length=64):
        super(TransportManager, self).__init__()
        self.__manager = Manager()
        self.channel_index = self.__manager.Value('i', channel_length)
        self.channel_data = self.__manager.list([0 for _ in range(channel_length)])
        self.channel_length = channel_length

        self.screen_process = ProcessScreen(self.channel_index, self.channel_data, self.channel_length)
        self.screen_process.start()
        # self.screen_process.join()

    def releaseResources(self):
        if self.screen_process.is_alive():
            self.screen_process.terminate()
            self.screen_process.join()
        self.__manager.shutdown()

    def gainTransData(self, data_count=1):
        index = self.channel_index.value
        data = []
        for count in range(data_count - 1, -1, -1):
            data.append(self.channel_data[(index - count) % self.channel_length])
        return data


if __name__ == '__main__':
    transport_manager = TransportManager()
    time.sleep(3)

    # 并行性验证
    for i in range(100000):
        print(i)
        print(transport_manager.channel_index.value)

    # 截图正确性验证
    while True:
        # 测试代码 查看窗口位置
        screen_grey = cv2.cvtColor(transport_manager.gainTransData()[0], cv2.COLOR_BGRA2BGR)
        cv2.imshow("window_main", screen_grey)
        cv2.moveWindow("window_main", 1000, 600)
        if cv2.waitKey(1) & 0xFF == ord('a'):
            print(cv2.waitKey(1))
            break
    cv2.waitKey()  # 视频结束后，按任意键退出
    cv2.destroyAllWindows()

    transport_manager.releaseResources()
