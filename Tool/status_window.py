from dependencies import *


class GlobalStatus:
    def __init__(self):
        self.start_time = time.time()
        self.last_time = time.time()

        self.goon = True  # go on是否停止
        self.round_reward = 0  # 总的奖赏
        self.round_step = 0  # 平均操作数
        self.total_step = 0  # 每一局总操作数

    def resetStatus(self):
        self.goon = True
        self.round_reward = 0
        self.round_step = 0
        self.total_step = 0

        self.start_time = time.time()
        self.last_time = self.start_time


class StatusWindow:
    """
    此类展示项目运行时的各种状态。
    This class shows the various states of the project when it is running.
    """

    def __init__(self):
        pass

    def tk_show(self):
        win = tkinter.Tk(className='状态 Status')
        # win.attributes("-alpha", 0.5)  # 调节透明度
        win.overrideredirect(True)  # 去除标题栏
        win.attributes('-topmost', True)
        win.geometry('300x200+1600+0')  # 调节大小
        win.mainloop()  # 显示窗口

    def cmd_show(self):
        pass

    def print_show(self):
        pass


if __name__ == '__main__':
    d = StatusWindow()
    T = threading.Thread(target=d.tk_show, args=())
    T.start()
    # d.tk_show()
    while True:
        print("test")
