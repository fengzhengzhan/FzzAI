import os
import torch
import torch.nn as nn
import math
import time
import numpy as np

from config import *

'''
===================================== Actor Net ===================================== 
'''
class ABasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ABasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # print(out.shape)
        out += residual
        out = self.relu(out)

        return out


class AResNet(nn.Module):
    def __init__(self, block=ABasicBlock, layers=[2, 2, 2, 2], action_dim=1):
        super(AResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool3d((1, 8, 12), stride=1)
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, action_dim)
        self.dropoutone = nn.Dropout(DROPOUT_ONE)
        self.dropouttwo = nn.Dropout(DROPOUT_TWO)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.tanh = nn.Tanh()

        # 初始化参数
        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
        #     elif isinstance(m, nn.BatchNorm3d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.Linear):
        #         m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        # print("[D] x:", x)
        x = self.fc1(x)
        x = self.dropoutone(x)
        x = self.leakyrelu(x)
        x = self.fc2(x)
        x = self.dropouttwo(x)
        x = self.tanh(x)
        x = x*0.4

        return x


'''
===================================== Critic Net ===================================== 
'''
class CBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CBasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class CResNet(nn.Module):
    def __init__(self, block=CBasicBlock, layers=[2, 2, 2, 2], action_dim=1):
        super(CResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool3d((1, 8, 12), stride=1)
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 1)
        self.afc1 = nn.Linear(action_dim, 64)
        self.leakyrelu = nn.LeakyReLU(inplace=True)

        # 初始化参数
        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
        #     elif isinstance(m, nn.BatchNorm3d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, a):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        y = self.afc1(a)
        out = self.leakyrelu(x+y)
        out = self.fc2(out)

        return out


'''
===================================== DDPG ===================================== 
'''
class AbrainModelDDPG():
    def __init__(self):
        self.Actor_eval = AResNet(action_dim=ACTION_DIM).to(DEVICE)
        self.Actor_target = AResNet(action_dim=ACTION_DIM).to(DEVICE)
        self.Critic_eval = CResNet().to(DEVICE)
        self.Critic_target = CResNet().to(DEVICE)

        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr=LR_A)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(), lr=LR_C)
        self.loss_td = nn.MSELoss()

        self.mu = MU
        self.sigma = SIGMA

        if not os.path.exists(MODEL_FILENAME):
            os.mkdir(MODEL_FILENAME)

    def choose_action(self, state, num_step):
        duration_s = self.Actor_eval(state)
        duration_s = duration_s[0][0].detach().cpu()
        print("网络预测动作 {}".format(duration_s), end=" ")
        duration_s = round(float(duration_s), 2)  # 保留两位小数

        # 增加扰动
        # if duration_s > 0:
        #     duration_s = duration_s - self.mu
        # elif duration_s < 0:
        #     duration_s = duration_s + self.mu
        duration_s = round(float(np.clip(np.random.normal(duration_s, self.sigma), -0.4, 0.4)), 2)
        # if self.mu > 0.0 or self.sigma > 0.0:
        if self.sigma > 0.0:
            # self.mu = MU_RATE - MU_RATE * num_step / RANDOM_EPISODE
            self.sigma = SIGMA - SIGMA * num_step / RANDOM_EPISODE

        return duration_s


    # 经验回放
    def train_network(self, replay, lossprintflag, num_step, batch_size=BATCH_SIZE):
        # 软更新网络参数
        for target_param, param in zip(self.Actor_target.parameters(), self.Actor_eval.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)
        for target_param, param in zip(self.Critic_target.parameters(), self.Critic_eval.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

        # step 1: 从 replay memory 随机采样  # TODO 线程sumtree!
        state_batch, action_batch, reward_batch, next_state_batch = replay.get(batch_size)
        # print(state_batch.shape, action_batch.shape, reward_batch.shape, next_state_batch.shape)
        # print("train_value:", action_batch[0][0], reward_batch[0][0])

        # ===================================动作网络
        a = self.Actor_eval(state_batch)
        q = self.Critic_eval(state_batch, a)  # loss=-q=-ce(s, ae(s))  更新ae ae(s)=a ae(s_)=a_
        loss_a = -torch.mean(q)  # 如果a是正确行为，那么它的Q更贴近0

        self.atrain.zero_grad()
        loss_a.backward()
        # torch.nn.utils.clip_grad_norm_(self.Actor_eval.parameters(), MAX_FORWORD)
        self.atrain.step()

        # =====================================策略网络
        a_ = self.Actor_target(next_state_batch)  # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        q_ = self.Critic_target(next_state_batch, a_)  # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        q_target = reward_batch + GAMMA * q_  # q_target = 负的
        # q_target = reward_batch  # q_target = 负的

        q_predict = self.Critic_eval(state_batch, action_batch)
        mls_loss = self.loss_td(q_target, q_predict)  # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba) 更新ce ,但这个ae(s)是记忆中的ba，让ce得出的Q靠近Q_target,让评价更准确

        self.ctrain.zero_grad()
        mls_loss.backward()
        self.ctrain.step()

        if lossprintflag:
            print("[*] Loss:", loss_a, mls_loss)


    def action_judge(self, gamescore):
        reward = round(float(int(gamescore) / 60), 2)
        if STORAGE_TARGET >= reward:
            reward = -round(float(math.sqrt(STORAGE_TARGET - reward)), 2)
        return reward

    def save(self):
        print("[-] Model saving...")
        torch.save(self.Actor_eval.state_dict(), MODEL_PATH + str("Actor_eval.pth"))
        torch.save(self.Actor_target.state_dict(), MODEL_PATH + str("Actor_target.pth"))
        torch.save(self.Critic_eval.state_dict(), MODEL_PATH + str("Critic_eval.pth"))
        torch.save(self.Critic_target.state_dict(), MODEL_PATH + str("Critic_target.pth"))
        print("[+] Model finish save!")

    def save_excellent(self, label=None):
        label = str(label) + "_" + str(time.strftime("%Y%m%d%H%M%S", time.localtime())) + "_excellent"
        print("[-] Model " + str(label) + "saving...")
        torch.save(self.Actor_eval.state_dict(), MODEL_PATH + str(str(label) + "Actor_eval.pth"))
        torch.save(self.Actor_target.state_dict(), MODEL_PATH + str(str(label) + "Actor_target.pth"))
        torch.save(self.Critic_eval.state_dict(), MODEL_PATH + str(str(label) + "Critic_eval.pth"))
        torch.save(self.Critic_target.state_dict(), MODEL_PATH + str(str(label) + "Critic_target.pth"))
        print("[-] Model " + str(label) + "finish save!")

    def load(self):
        if os.path.exists(MODEL_PATH + str("Actor_eval.pth")) \
                and os.path.exists(MODEL_PATH + str("Actor_target.pth")) \
                and os.path.exists(MODEL_PATH + str("Critic_eval.pth")) \
                and os.path.exists(MODEL_PATH + str("Critic_target.pth")):
            self.Actor_eval.load_state_dict(torch.load(MODEL_PATH + str("Actor_eval.pth")))
            self.Actor_target.load_state_dict(torch.load(MODEL_PATH + str("Actor_target.pth")))
            self.Critic_eval.load_state_dict(torch.load(MODEL_PATH + str("Critic_eval.pth")))
            self.Critic_target.load_state_dict(torch.load(MODEL_PATH + str("Critic_target.pth")))

            print("[+] model load finish!")



if __name__ == '__main__':
    AbrainModelDDPG()