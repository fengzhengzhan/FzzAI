import torch
import os

# Abrain_model
ACTION_DIM = 1
MODEL_FILENAME = "SuperHexagonModel"

KERNEL_3D = (1, 8, 12)
STORAGE_TARGET = 4  # 阶段性目标，为使reward为负

LR_A = 0.000001
LR_C = 0.000002
GAMMA = 0.9
TAU = 0.01  # 软更新

DROPOUT_ONE = 0.2
DROPOUT_TWO = 0.5
SCALE_VALUE = 0.4

MU = 0.6
MU_RATE = 0.995
SIGMA = 0.8
SIGMA_RATE = 0.996
RANDOM_EPISODE = 1000
BATCH_SIZE = 4

MODEL_PATH = MODEL_FILENAME + "\\"


# Aeye_grabscreen
SUPERHEXAGON_WINDOW = (0, 32, 768, 508)
MANAGER_LIST_LENGTH = 16
RESIZE_WINDOW = (384, 238)  # (WIDTH, HEIGHT)
SLEEP_SCREEN = 0.02  # 每秒帧数，同步帧率   (0.02, 0.16)
SLEEP_INIT_GRAB = 2  # 先开启截取屏幕进程，并等待时间填充满缓冲数组


# Amemory_replaybuffer
REPLAY_SIZE = 240  # 经验回放池大小
POSITIVEMODE = False  # 是否存储reward为正的情况
REPLAY_POSSIZE = 140  # 正例(reward>0)经验池大小

SUPERHEXAGON_STORE_PATH = MODEL_FILENAME + "\\superhexagon_store_path"
SUPERHEXAGON_POSSTORE_PATH = MODEL_FILENAME + "\\superhexagon_posstore_path"
SUPERHEXAGON_REWARDFLAG_PATH = MODEL_FILENAME + "\\superhexagon_rewardflag_path"


# config
ROLE_NAME = "SH004"
CPUDEVICE = "cpu"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GAME_HANDLE = "Super Hexagon"


# main_traintest
TEST_MODE = False  # 默认是训练模式
MAX_WORKERS = 16
EPISODES = 30000
END_GAME_TIME = 0
GLOBAL_BEST_REWARD = 10

TRAINSTORAGELEN = 160  # 200训练条件: 回放池超过数量 #TODO
TRAINEPISODELEN = 2  # 训练条件: 轮数 #TODO
TRAININGDURATION = 3  # 持续时间 s

SLEEP_GAME = 2
SAVE_FIRST_EPISODE = 1
SAVE_EPISODE = 30

DEBUG_MODE = False
if DEBUG_MODE:
    TRAINSTORAGELEN = 10
    TRAINEPISODELEN = 2

VALIDATION_EXPERIMENT = True
if VALIDATION_EXPERIMENT:
    SCALE_VALUE = 2
    LR_A = 0.000001
    LR_C = 0.000002
    GAMMA = 0.9
    TAU = 0.01
    BATCH_SIZE = 16
    RENDER = True
    ENV_NAME = "Pendulum-v0"
    TRAININGDURATION = 10  # 持续时间 s

    SUPERHEXAGON_WINDOW = (50, 100, 449, 499)
    RESIZE_WINDOW = (200, 200)
    KERNEL_3D = (1, 7, 7)

    MODEL_FILENAME = ENV_NAME + "_Model"
    if not os.path.exists(MODEL_FILENAME):
        os.mkdir(MODEL_FILENAME)
    MODEL_PATH = MODEL_FILENAME + "\\"
    SUPERHEXAGON_STORE_PATH = MODEL_FILENAME + "\\validation_store_path"
    SUPERHEXAGON_POSSTORE_PATH = MODEL_FILENAME + "\\validation_posstore_path"
    SUPERHEXAGON_REWARDFLAG_PATH = MODEL_FILENAME + "\\validation_rewardflag_path"

    SIGMA = 3
    MAX_EP_STEPS = 200
    REPLAY_SIZE = 500
