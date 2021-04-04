import time
import random
from argconfig import *
import pickle
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T

# 将REPLAY_BUFFER经验回放的存储库弹出一部分
REPLAY_BUFFER = []
if os.path.exists(DQN_STORE_PATH):
    REPLAY_BUFFER = pickle.load(open(DQN_STORE_PATH, 'rb'))
for i in range(600):
    REPLAY_BUFFER.pop(len(REPLAY_BUFFER)-1)
pickle.dump(REPLAY_BUFFER, open(DQN_STORE_PATH, 'wb'))
print(REPLAY_BUFFER, type(REPLAY_BUFFER), len(REPLAY_BUFFER))
