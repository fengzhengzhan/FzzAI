import torch
from collections import deque
import time

# DQN arguments


# small_BATCH_SIZE = 16
# big_BATCH_SIZE = 128
# BATCH_SIZE_door = 1000

GAMMA = 0.9
LEARN_RATE = 0.001
MOVE_LEARN_RATE = 0.001
# FINAL_EPSILON = 0.00002
# EPSILON = 0.35

# 如果使用gpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIONATTACK_SIZE = 7
ACTIONMOVE_SIZE = 7

# train
# WIDTH = 290  # 290
# HEIGHT = 128  # 128
WIDTH = 236  # 236
HEIGHT = 122  # 122

# main_window = (66, 99, 1225, 610)  # (528, 1160, 4) -> (128, 290) 4倍缩放
main_window = (178, 99, 1121, 626)  # (904, 488) => (236, 122)
boss_blood_window = (150, 660, 1140, 680)
blood_window = (227, 101, 530, 102)
power_window = (176, 93, 177, 158)

DQN_MODEL_PATH = "E:\\1_mycode\\Knight_DQN\\model\\dqn_model.pt"
DQN_MODELMOVE_PATH = "E:\\1_mycode\\Knight_DQN\\model\\dqn_movemodel.pt"
DQN_STORE_PATH = "E:\\1_mycode\\Knight_DQN\\model\\dqn_store"
DQN_TREE_PATH = "E:\\1_mycode\\Knight_DQN\\model\\dqn_tree.npy"
DQN_LOG_PATH = "model\\dqn_log.log"

EPISODES = 30000
REPLAY_SIZE = 1000
STORE_SIZE = 4  # 所需要存储的次数 1500
BATCH_SIZE = 2  # 采样个数
ALL_BATCH_SIZE = 64  # 储存满后疯狂经验回放
UPDATE_STEP = 120  # 更新target_net次数
ONE_ATTACK = 30

TRAINING_NUM = 140
TOTAL_NUM = 200

PRIO_MAX = 0.1
A = 0.6
E = 0.01


CHOOSE_ACTION_TIME = 60.0

NUM_STEP = 0
TARGET_STEP = 0

paused = True
init_time = time.time()

num_step = 0
target_step = 0



I_ATTACK = 0
K_ATTACK = 1
ATTACK_DOUBLE_NUM = 2
ATTACK_NUM = 3
Q_SHORT_USE = 4
IQ_BOOM = 5
KQ_BOOM = 6
Q_LONG_USE = 7

J_LEFT = 0
L_RIGHT = 1
E_RUN = 2
TO_LEFT = 4
TO_RIGHT = 5
SPACE_STAY = 6

BOSS_ALL_BLOOD = 950

# 持续时间为reward  费血 -100 回血 +100 攻击 +0.5 增加能量 +10

# REDUCE_BLOOD = -100

Layers = [2, 2, 2, 2]