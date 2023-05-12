import os
import Tool.project_path as projectpath

GAME_NAME = "HollowKnight"

# Save Path
PATH_DATASET = GAME_NAME + "Data"

# Network arguments
GAMMA = 0.9
LEARN_RATE = 0.001

# Train arguments
EPOCH = 30000

ACTION_SIZE = 10

# train
WIDTH = 290  # 290 获取屏幕图片后缩放比例
HEIGHT = 128  # 128

main_window = (66, 99, 1225, 610)  # (1160, 512, 4) -> (128, 290) 4倍缩放
blood_window = (227, 97, 530, 98)
power_window = (176, 93, 177, 158)

DQN_MODEL_PATH = projectpath.dateset_autosave_path + os.sep + PATH_DATASET + os.sep + "DQN_model.pt"
DQN_STORE_PATH = "E:\\1_mycode\\Knight_DQN\\model\\dqn_store"
DQN_LOG_PATH = "model\\dqn_log.log"

REPLAY_SIZE = 4000
STORE_SIZE = 1200  # 所需要存储的次数 1500
BATCH_SIZE = 8  # 采样个数
UPDATE_STEP = 50  # 更新target_net次数

CHOOSE_ACTION_TIME = 30.0

NUM_STEP = 0
TARGET_STEP = 0

paused = True
# init_time = time.time()

num_step = 0
target_step = 0

I_ATTACK = 0
J_LEFT = 1
K_ATTACK = 2
L_RIGHT = 3
ATTACK_NUM = 4
