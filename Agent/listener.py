from dependencies import *


class ProcessListenerKeyboard(Process):
    def __init__(self):
        super(ProcessListenerKeyboard, self).__init__()
        self.channel_length = None
        self.channel_data = None
        self.channel_index = None

        self.map_keycode = {
            'ctrl_l': keyboard.Key.ctrl_l, 'ctrl_r': keyboard.Key.ctrl_r,
            'p': keyboard.KeyCode.from_char('p'), 's': keyboard.KeyCode.from_char('p'),
            'ctrl_l+p': keyboard.KeyCode.from_char('\x10'), 'ctrl_l+s': keyboard.KeyCode.from_char('\x13'),
        }

    def setChannel(self, channel_index, channel_data, channel_length):
        self.channel_index = channel_index
        self.channel_data = channel_data
        self.channel_length = channel_length

    def gainKey(self, key):
        # 滚动Manager数组
        self.channel_data[(self.channel_index.value + 1) % self.channel_length] = key
        self.channel_index.value += 1

    def onPress(self, key):
        self.gainKey(key)

    def onRelease(self, key):
        self.gainKey(key)

    def run(self):
        with keyboard.Listener(on_press=self.onPress, on_release=self.onRelease) as listener:
            listener.join()


class ProcessListenerMouese(Process):
    def __init__(self, channel_index, channel_data, channel_length):
        super(ProcessListenerMouese, self).__init__()
        self.channel_index = channel_index
        self.channel_data = channel_data
        self.channel_length = channel_length

    def gainKey(self, key):
        # 滚动Manager数组
        self.channel_data[(self.channel_index.value + 1) % self.channel_length] = key
        self.channel_index.value += 1

    def onMove(self, x, y):
        self.gainKey(("move", x, y, x, y))

    def onClick(self, x, y, button, pressed):
        self.gainKey(("click", x, y, button, pressed))

    def onScroll(self, x, y, dx, dy):
        self.gainKey(("scroll", x, y, dx, dy))

    def run(self):
        with mouse.Listener(on_move=self.onMove, on_click=self.onClick, on_scroll=self.onScroll) as listener:
            listener.join()


import time
from pynput import keyboard

if __name__ == "__main__":
    # time.sleep(3)
    plk = ProcessListenerKeyboard()
    transport_manager = TransportManager(plk)
    map_keycode = plk.map_keycode

    # 并行性验证
    for i in range(100000):
        time.sleep(1)
        print(i)
        print(transport_manager.channel_index.value)
        print(transport_manager.channel_data)
        if map_keycode['ctrl_l+p'] in transport_manager.channel_data:
            print("pause")
            if map_keycode['ctrl_l+s'] in transport_manager.channel_data:
                print('start')

    # transport_manager = TransportManager(ProcessListenerMouese)
    # # 并行性验证
    # for i in range(100000):
    #     time.sleep(1)
    #     print(i)
    #     print(transport_manager.channel_index.value)
    #     print(transport_manager.channel_data)
