from dependencies import *

class ListenerKeyboard:
    def onPress(self, key):
        pass


class ListenerMouse:
    pass


class ProcessListenerKeyboard:
    pass


class ProcessListenerMouese:
    pass


import time
from pynput import keyboard


class GlobalKeyboardListener:
    def __init__(self):
        self.listener = None

    def on_press(self, key):
        try:
            print(f"Key {key.char} pressed")
        except AttributeError:
            print(f"Special key {key} pressed")

    def on_release(self, key):
        print(f"Key {key} released")
        if key == keyboard.Key.esc:
            return False

    def start_listener(self):
        with keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release) as listener:
            self.listener = listener
            listener.join()


if __name__ == "__main__":
    # time.sleep(3)
    gkl = GlobalKeyboardListener()
    gkl.start_listener()
    # gkl_thread = threading.Thread(target=gkl.start_listener)
    # gkl_thread.start()
    # time.sleep(10)  # 主进程阻塞1秒
    # gkl.listener.stop()
    # gkl_thread.join()

    while True:
        time.sleep(1)
        print(10)



#