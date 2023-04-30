from dependencies import *

class StatusWindow(object):
    """
    此类展示项目运行时的各种状态。
    This class shows the various states of the project when it is running.
    """
    def __init__(self):
        pass

    def tk_show(self):
        win = Tk(className='状态 Status')
        win.attributes("-alpha", 0.5)  # 调节透明度
        # root.overrideredirect(True)  # 去除标题栏
        win.attributes('-topmost', True)
        win.geometry('300x200+20+40')  # 调节大小
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
