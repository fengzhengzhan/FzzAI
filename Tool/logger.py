from dependencies import *


class Logger:
    """
    [00:00:00][+]message
    [+]正确进度
    [-]删除操作

    [*]debug
    [i]info
    [?]warning
    [!]error
    [!!]critical
    """

    def __init__(self, log_path, level=logging.DEBUG):
        # 初始化日志等级
        self.level = level

        # %(asctime)s 格式为 %Y-%m-%d %H:%M:%S,%f  三位毫秒数值
        # LOG_FORMAT = "%(asctime)s - %(filename)s - %(levelname)s - %(message)s"
        # datefmt='%Y-%m-%d %H:%M:%S.%f'
        LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            filename=log_path,
            filemode='a',
            level=self.level,
            format=LOG_FORMAT
        )

    def log(self, msg):
        """记录日志"""
        logging.info(msg)

    def printControl(self):
        pass

    def printStateWindow(self):
        pass

    def setLevel(self, level):
        """设置日志等级"""
        pass


if __name__ == '__main__':
    logger = Logger("C:\Files\Data\Life\Code\GithubCode\FzzAI\dataset_autosave" + os.sep + confglobal.GLOBAL_LOG_PATH)
    logger.log("test")
