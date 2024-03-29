import os
from logging import *

import Tool.project_path as projectpath
import conf as confglobal

"""
[00:00:00][+]message
[+]创建操作
[-]删除操作

[*]debug
[i]info
[?]warning
[!]error
[!!]critical
"""

# 初始化日志等级
# %(asctime)s 格式为 %Y-%m-%d %H:%M:%S,%f  三位毫秒数值
# LOG_FORMAT = "%(asctime)s - %(filename)s - %(levelname)s - %(message)s"
# datefmt='%Y-%m-%d %H:%M:%S.%f'
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
basicConfig(
    filename=projectpath.dateset_autosave_path + os.sep + confglobal.GLOBAL_LOG_PATH,
    filemode='a',
    level=DEBUG,
    format=LOG_FORMAT,
    encoding='utf-8',
)


# print(projectpath.dateset_autosave_path + os.sep + confglobal.GLOBAL_LOG_PATH)

def logFile(level, msg):
    """将日志输出到日志文件"""
    if level == DEBUG:
        debug("[*] " + msg)
    elif level == INFO:
        info("[i] " + msg)
    elif level == WARNING:
        warning("[?] " + msg)
    elif level == ERROR:
        error("[!] " + msg)
        raise Exception("[!] " + msg)
    elif level == CRITICAL:
        critical("[!!] " + msg)
        raise Exception("[!!] " + msg)


def logControl(level, msg):
    """将日志输出到控制台"""
    printflag = ""
    if level == DEBUG:
        printflag = "[*] "
    elif level == INFO:
        printflag = "[i] "
    elif level == WARNING:
        printflag = "[?] "
    elif level == ERROR:
        printflag = "[!] "
    elif level == CRITICAL:
        printflag = "[!!] "
    print(printflag + msg)


def logStatusWindow(level, msg):
    """将日志输出到状态窗口"""
    pass


def projlog(level, msg):
    """记录日志"""
    msg = str(msg)
    logFile(level, msg)
    if confglobal.HAS_LOG_CONTROL:
        logControl(level, msg)
    if confglobal.HAS_LOG_WINDOW:
        logStatusWindow(level, msg)
