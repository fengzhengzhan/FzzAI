import torch
import os

# Abrain_model
# 0.04秒可以识别动作
ACTION_STEPS = [-0.4, -0.39, -0.38, -0.37, -0.36, -0.35, -0.34, -0.33, -0.32, -0.31,
                -0.3, -0.29, -0.28, -0.27, -0.26, -0.25, -0.24, -0.23, -0.22, -0.21,
                -0.2, -0.19, -0.18, -0.17, -0.16, -0.15, -0.14, -0.13, -0.12, -0.11,
                -0.1, -0.09, -0.08, -0.07, -0.06, -0.05, -0.04, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19,
                0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29,
                0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4]
ACTION_DIM = 74
MODEL_FILENAME = "SuperHexagonModel"

KERNEL_SIZE = (14, 25)
STORAGE_TARGET = 4  # 阶段性目标，为使reward为负

LR = 0.0001
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
BATCH_SIZE = 12

FORWARD_SCALE = 0.8  # scale for loss function of forward prediction model, 0.8
INVERSE_SCALE = 0.2  # scale for loss function of inverse prediction model, 0.2
LOSSQ_SCALE = 1  # scale for loss function of Q value, 1

UPDATE_TIME = 20

MODEL_PATH = MODEL_FILENAME + "\\"
MODEL_WEIGHT_PATH = "efficientnetb0.pth"


# Aeye_grabscreen
SUPERHEXAGON_WINDOW = (1, 79, 768, 508)  # (0, 32, 768, 508)   384 238
MANAGER_LIST_LENGTH = 16
RESIZE_WINDOW = (384, 215)  # (WIDTH, HEIGHT)
SLEEP_SCREEN = 0.02  # 每秒帧数，同步帧率   (0.02, 0.16)
SLEEP_INIT_GRAB = 2  # 先开启截取屏幕进程，并等待时间填充满缓冲数组


# Amemory_replaybuffer
REPLAY_SIZE = 320  # 经验回放池大小
POSITIVEMODE = False  # 是否存储reward为正的情况
REPLAY_POSSIZE = 140  # 正例(reward>0)经验池大小

SUPERHEXAGON_STORE_PATH = MODEL_FILENAME + "\\superhexagon_store_path"
SUPERHEXAGON_POSSTORE_PATH = MODEL_FILENAME + "\\superhexagon_posstore_path"
SUPERHEXAGON_REWARDFLAG_PATH = MODEL_FILENAME + "\\superhexagon_rewardflag_path"


# config
ROLE_NAME = "SH010"
CPUDEVICE = "cpu"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GAME_HANDLE = "Super Hexagon"


# main_traintest
TEST_MODE = False  # 默认是训练模式
MAX_WORKERS = 16
EPISODES = 30000
END_GAME_TIME = 0
GLOBAL_BEST_REWARD = 10

TRAINSTORAGELEN = 260  # 200训练条件: 回放池超过数量 #TODO
TRAINEPISODELEN = 2  # 训练条件: 轮数 #TODO
TRAININGDURATION = 2  # 持续时间 s
TRAINPERNUM = 2  # 持续次数

SLEEP_BREAK = 0.04
SLEEP_GAME = 0.5
SLEEP_INTERVAL = 2
SAVE_FIRST_EPISODE = 1
SAVE_EPISODE = 30

DEBUG_MODE = True
if DEBUG_MODE:
    TRAINSTORAGELEN = 10
    TRAINEPISODELEN = 0

VALIDATION_EXPERIMENT = False
if VALIDATION_EXPERIMENT:
    SCALE_VALUE = 2
    LR_A = 0.0001
    LR_C = 0.0002
    GAMMA = 0.9
    TAU = 0.01
    BATCH_SIZE = 16
    RENDER = True
    ENV_NAME = "CartPole-v0"
    ACTION_DIM = 2
    TRAININGDURATION = 10  # 持续时间 s

    SUPERHEXAGON_WINDOW = (0, 40, 599, 423)
    RESIZE_WINDOW = (300, 192)
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
