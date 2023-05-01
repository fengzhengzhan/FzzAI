from dependencies import *

class KeyboardListener:
    def __init__(self, callback):
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.callback = callback

    def start(self):
        self.listener.start()

    def stop(self):
        self.listener.stop()

    def on_press(self, key):
        try:

            self.callback(key.char)
        except AttributeError:
            self.callback(str(key))

    def on_release(self, key):
        pass


if __name__ == '__main__':
    def my_callback(key):
        print(key)

    # listener = KeyboardListener(my_callback)
    # listener.start()
    while True:
        print(win32api.GetAsyncKeyState(key))