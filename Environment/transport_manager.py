# -*- coding: utf-8 -*-
from dependencies import *

class TransportManager:
    def __init__(self, ProcessClass, channel_length=64):
        super(TransportManager, self).__init__()
        self.__manager = Manager()
        self.channel_index = self.__manager.Value('i', channel_length)
        self.channel_data = self.__manager.list([0 for _ in range(channel_length)])
        self.channel_length = channel_length

        self.screen_process = ProcessClass(self.channel_index, self.channel_data, self.channel_length)
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