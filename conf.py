import torch

# Path of Project
ROOTDIR = "FzzAI"
DATASET_AUTOSAVE_PATH = "dataset_autosave"
DATASET_SQLITE = "sqlitedb"

# Log
GLOBAL_LOG_PATH = "global.log"
HAS_LOG_CONTROL = True
HAS_LOG_WINDOW = True

# 如果使用gpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# agent_sense

# Vision
# 枚举传入的视觉数组类型的序列号。
# Enumerate the sequence number of the incoming visual array type.
VIS_IDX_NUM = 0 # 所取得的图片数量
VIS_IDX_HANDLE = 1  # 窗口句柄
VIS_IDX_SIZE = 2  # 所获得的视觉窗口的尺寸大小()
VIS_IDX_MODE = 3  # 如图片相减之类的方式
VIS_IDX_FUNC = 4  # 使用函数所规定的方式处理图片


