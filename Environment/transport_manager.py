# -*- coding: utf-8 -*-
from dependencies import *


class TransportManager:
    def __init__(self, manager, channel_length=64):
        super(TransportManager, self).__init__()
        self.channel_index = manager.Value('i', 0)
        self.channel_data = manager.list([0 for _ in range(channel_length)])
        self.channel_length = channel_length

    def sendTransData(self, data):
        # 滚动Manager数组
        self.channel_data[(self.channel_index.value + 1) % self.channel_length] = data
        self.channel_index.value += 1

    def gainTransData(self, data_count=1):
        index = self.channel_index.value
        data = []
        for count in range(data_count - 1, -1, -1):
            data.append(self.channel_data[(index - count) % self.channel_length])
        return data

    def clearTransData(self):
        """Manager内容清空，设置为0初始值"""
        for i in range(self.channel_length):
            self.channel_data[i] = 0
        self.channel_index.value = 0


class ProcessTransportManager:
    def __init__(self, process_channel):
        """
        process_channel 传入类对象。继承Process，添加setManager()方法
        """
        # Manager从主进程传入，用于进程间数据通讯
        self.__manager = Manager()
        self.transport_manager = TransportManager(self.__manager)
        self.process = process_channel
        self.process.setManager(self.transport_manager)
        self.process.start()  # 创建新进程，新进程会调用run()方法。
        # self.process.join()

    def releaseResources(self):
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
        self.__manager.shutdown()
