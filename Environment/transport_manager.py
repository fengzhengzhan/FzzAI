# -*- coding: utf-8 -*-
from dependencies import *

class TransportManager:
    def __init__(self, process_channel, channel_length=64):
        super(TransportManager, self).__init__()
        self.__manager = Manager()
        self.channel_index = self.__manager.Value('i', 0)
        self.channel_data = self.__manager.list([0 for _ in range(channel_length)])
        self.channel_length = channel_length

        self.process = process_channel
        self.process.setChannel(self.channel_index, self.channel_data, self.channel_length)
        self.process.start()  # 创建新进程，新进程会调用run()方法。
        # self.screen_process.join()

    def releaseResources(self):
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
        self.__manager.shutdown()

    def gainTransData(self, data_count=1):
        index = self.channel_index.value
        data = []
        for count in range(data_count - 1, -1, -1):
            data.append(self.channel_data[(index - count) % self.channel_length])
        return data