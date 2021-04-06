import torch
from collections import deque
import time

# DQN arguments
GAMMA = 0.9
LR_MOVE = 0.001  # actor学习率
LR_ATTACK = 0.001  # critic学习率
TAU = 0.05  # 软更新

MAX_FORWORD = 10  # 传播clip

# 如果使用gpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIONATTACK_SIZE = 7
ACTIONMOVE_SIZE = 7

# train
main_window = (178, 99, 1121, 626)  # (904, 488) => (236, 122)
boss_blood_window = (150, 660, 1140, 680)
blood_window = (224, 101, 530, 150)
power_window = (176, 93, 177, 158)

DQN_MODEL_PATH = "E:\\1_mycode\\Knight_DQN\\model\\"
DQN_STORE_PATH = "E:\\1_mycode\\Knight_DQN\\model\\dqn_store"
DQN_POSSTORE_PATH = "E:\\1_mycode\\Knight_DQN\\model\\dqn_posstore"
DQN_LOG_PATH = "model\\dqn_log.log"


EPISODES = 30000
REPLAY_SIZE = 1000
STORE_SIZE = 600  # 所需要存储的次数 1500
REPLAY_POSSIZE = 600
STORE_POSSIZE = 300
BATCH_SIZE = 2  # 采样个数
ALL_BATCH_SIZE = 64  # 储存满后疯狂经验回放
UPDATE_STEP = 120  # 更新target_net次数
ONE_ATTACK = 30

JUDGE_DONE = 10  # 死亡血量为0判断

TRAINING_NUM = 150
TOTAL_NUM = 200


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
SPACE_SHORT = 3
TO_LEFT = 4
TO_RIGHT = 5
SPACE_STAY = 6

SELF_BLOOD = 9
BOSS_ALL_BLOOD = 950

ch_attack_action = ['上击', '下击', '攻击', '双击', '放波', '上吼', '下砸', ]
ch_move_action = ['左走', '右走', '冲刺', '短跳', '向左', '向右', '长跳', ]

# 持续时间为reward  费血 -100 回血 +100 攻击 +0.5 增加能量 +10

# REDUCE_BLOOD = -100
