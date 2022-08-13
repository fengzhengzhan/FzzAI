import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

import math
import time
import copy
import numpy as np
import random

from functools import partial
from collections import OrderedDict
from typing import Optional, Callable

from config import *
from Abody_keyboard import sample_select_action


# Noisy_DQN
class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise
    N.B. nn.Linear already initializes weight and bias to
    """
    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))

    def forward(self, input):
        bias = self.bias
        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))

        with torch.no_grad():
            torch.randn(self.epsilon_input.size(), out=self.epsilon_input)
            torch.randn(self.epsilon_output.size(), out=self.epsilon_output)
            eps_in = func(self.epsilon_input)
            eps_out = func(self.epsilon_output)
            noise_v = torch.mul(eps_in, eps_out).detach()
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)

# efficientnet_b0
def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf

    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)

        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),
                                               norm_layer(out_planes),
                                               activation_layer())


class SqueezeExcitation(nn.Module):
    def __init__(self,
                 input_c: int,   # block input channel
                 expand_c: int,  # block expand channel
                 squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = input_c // squeeze_factor
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
        self.ac1 = nn.SiLU()  # alias Swish
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x


class InvertedResidualConfig:
    # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate
    def __init__(self,
                 kernel: int,          # 3 or 5
                 input_c: int,
                 out_c: int,
                 expanded_ratio: int,  # 1 or 6
                 stride: int,          # 1 or 2
                 use_se: bool,         # True
                 drop_rate: float,
                 index: str,           # 1a, 2a, 2b, ...
                 width_coefficient: float):
        self.input_c = self.adjust_channels(input_c, width_coefficient)
        self.kernel = kernel
        self.expanded_c = self.input_c * expanded_ratio
        self.out_c = self.adjust_channels(out_c, width_coefficient)
        self.use_se = use_se
        self.stride = stride
        self.drop_rate = drop_rate
        self.index = index

    @staticmethod
    def adjust_channels(channels: int, width_coefficient: float):
        return _make_divisible(channels * width_coefficient, 8)


class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        layers = OrderedDict()
        activation_layer = nn.SiLU  # alias Swish

        # expand
        if cnf.expanded_c != cnf.input_c:
            layers.update({"expand_conv": ConvBNActivation(cnf.input_c,
                                                           cnf.expanded_c,
                                                           kernel_size=1,
                                                           norm_layer=norm_layer,
                                                           activation_layer=activation_layer)})

        # depthwise
        layers.update({"dwconv": ConvBNActivation(cnf.expanded_c,
                                                  cnf.expanded_c,
                                                  kernel_size=cnf.kernel,
                                                  stride=cnf.stride,
                                                  groups=cnf.expanded_c,
                                                  norm_layer=norm_layer,
                                                  activation_layer=activation_layer)})

        if cnf.use_se:
            layers.update({"se": SqueezeExcitation(cnf.input_c,
                                                   cnf.expanded_c)})

        # project
        layers.update({"project_conv": ConvBNActivation(cnf.expanded_c,
                                                        cnf.out_c,
                                                        kernel_size=1,
                                                        norm_layer=norm_layer,
                                                        activation_layer=nn.Identity)})

        self.block = nn.Sequential(layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

        # 只有在使用shortcut连接时才使用dropout层
        if self.use_res_connect and cnf.drop_rate > 0:
            self.dropout = DropPath(cnf.drop_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        result = self.dropout(result)
        if self.use_res_connect:
            result += x

        return result


class EfficientNet(nn.Module):
    def __init__(self,
                 width_coefficient: float,
                 depth_coefficient: float,
                 num_classes: int = 1000,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None
                 ):
        super(EfficientNet, self).__init__()

        # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate, repeats
        default_cnf = [[3, 32, 16, 1, 1, True, drop_connect_rate, 1],
                       [3, 16, 24, 6, 2, True, drop_connect_rate, 2],
                       [5, 24, 40, 6, 2, True, drop_connect_rate, 2],
                       [3, 40, 80, 6, 2, True, drop_connect_rate, 3],
                       [5, 80, 112, 6, 1, True, drop_connect_rate, 3],
                       [5, 112, 192, 6, 2, True, drop_connect_rate, 4],
                       [3, 192, 320, 6, 1, True, drop_connect_rate, 1]]

        def round_repeats(repeats):
            """Round number of repeats based on depth multiplier."""
            return int(math.ceil(depth_coefficient * repeats))

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        adjust_channels = partial(InvertedResidualConfig.adjust_channels,
                                  width_coefficient=width_coefficient)

        # build inverted_residual_setting
        bneck_conf = partial(InvertedResidualConfig,
                             width_coefficient=width_coefficient)

        b = 0
        num_blocks = float(sum(round_repeats(i[-1]) for i in default_cnf))
        inverted_residual_setting = []
        for stage, args in enumerate(default_cnf):
            cnf = copy.copy(args)
            for i in range(round_repeats(cnf.pop(-1))):
                if i > 0:
                    # strides equal 1 except first cnf
                    cnf[-3] = 1  # strides
                    cnf[1] = cnf[2]  # input_channel equal output_channel

                cnf[-1] = args[-2] * b / num_blocks  # update dropout ratio
                index = str(stage + 1) + chr(i + 97)  # 1a, 2a, 2b, ...
                inverted_residual_setting.append(bneck_conf(*cnf, index))
                b += 1

        # create layers
        layers = OrderedDict()

        # first conv
        layers.update({"stem_conv": ConvBNActivation(in_planes=1,
                                                     out_planes=adjust_channels(32),
                                                     kernel_size=3,
                                                     stride=2,
                                                     norm_layer=norm_layer)})

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.update({cnf.index: block(cnf, norm_layer)})

        # build top
        last_conv_input_c = inverted_residual_setting[-1].out_c
        last_conv_output_c = adjust_channels(1280)
        layers.update({"top": ConvBNActivation(in_planes=last_conv_input_c,
                                               out_planes=last_conv_output_c,
                                               kernel_size=1,
                                               norm_layer=norm_layer)})

        self.features = nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        classifier = []
        if dropout_rate > 0:
            classifier.append(nn.Dropout(p=dropout_rate, inplace=True))
        classifier.append(nn.Linear(last_conv_output_c, num_classes))
        self.classifier = nn.Sequential(*classifier)

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)



def efficientnet_b0(num_classes=ACTION_DIM):
    # input image size 224x224
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.0,
                        dropout_rate=0.2,
                        num_classes=num_classes)

'''
===================================== DQN ===================================== 
'''
class AbrainModel():
    def __init__(self):
        self.DQN_eval = efficientnet_b0(num_classes=ACTION_DIM).to(DEVICE)
        self.DQN_target = efficientnet_b0(num_classes=ACTION_DIM).to(DEVICE)
        # self.DQN_target.load_state_dict(self.DQN_eval.state_dict())
        # parm = {}
        # for name, parameters in self.DQN_eval.named_parameters():
        #     print(name, ':', parameters.size())
        #     parm[name] = parameters.detach().cpu().numpy()
        #     print(parm[name])
        # print(self.DQN_eval)

        self.train = torch.optim.Adam(self.DQN_eval.parameters(), lr=LR)
        self.loss_td = nn.MSELoss()

        if not os.path.exists(MODEL_FILENAME):
            os.mkdir(MODEL_FILENAME)


    def choose_action(self, state, pro_step_num, step_num):
        mutetu = pro_step_num / (step_num * 2)
        if random.random() >= mutetu:
            # duration_num = random.randint(0, ACTION_DIM-1)
            # print("随机位置:", duration_num, end=" ")

            duration_num = self.DQN_eval(state)
            # print(duration_num)
            duration_num = np.array(duration_num.detach().cpu()[0])
            duration_num = sample_select_action(duration_num)
            # duration_num = int(duration_num.argmax())
            print("采样:", duration_num, end=" ")
        else:
            duration_num = self.DQN_eval(state)
            # print(duration_num)
            duration_num = np.array(duration_num.detach().cpu()[0])
            # duration_num = sample_select_action(duration_num)
            duration_num = int(duration_num.argmax())
            print("预测位置:", duration_num, end=" ")
        return duration_num


    # 经验回放
    def train_network(self, replay, lossprintflag, num_step, per_step, batch_size=BATCH_SIZE):
        # 软更新网络参数
        for target_param, param in zip(self.DQN_target.parameters(), self.DQN_eval.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)
        # 直接更新
        # if num_step % UPDATE_TIME == 0 and per_step == 1:
        #     self.DQN_target.load_state_dict(self.DQN_eval.state_dict())

        # step 1: 从 replay memory 随机采样  # TODO 线程sumtree!
        state_batch, action_batch, reward_batch, next_state_batch = replay.get(batch_size)
        # print(state_batch.shape, action_batch, reward_batch, next_state_batch.shape)
        # print("train_value:", action_batch[0][0], reward_batch[0][0])

        # ===================================动作网络
        q = self.DQN_eval(state_batch).gather(1, action_batch)
        q_next = self.DQN_target(next_state_batch).detach().max(1)[0].reshape(-1, 1)
        tq = reward_batch + GAMMA * q_next
        # tq = reward_batch
        # print(q, tq, reward_batch)

        loss = self.loss_td(q, tq)

        self.train.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.DQN_eval.parameters(), 10)
        self.train.step()

        if lossprintflag:
            print("[*] Loss:", loss)


    def action_judge(self, gamescore, pro_action_num, action_num, area, next_area):
        # print(area, next_area)

        reward = round(float(int(gamescore) / 60), 2)

        if len(area[1]) != 0 and len(next_area[1]) != 0 \
                and len(area[2]) != 0 and len(next_area[2]) != 0:
            cid = 3
            # 先右再左
            if 0.1 <= area[0][3] <= 0.3:
                cid = 3
            elif 0.1 <= area[0][4] <= 0.3:
                cid = 4
            elif 0.1 <= area[0][2] <= 0.3:
                cid = 2
            elif 0.1 <= area[0][5] <= 0.3:
                cid = 5
            elif 0.1 <= area[0][1] <= 0.3:
                cid = 1
            elif 0.1 <= area[0][0] <= 0.3:
                cid = 0
            first_distance = math.sqrt( (area[1][cid][0] - area[2][0])**2 + (area[1][cid][1] - area[2][1])**2 )

            ncid = 3
            if 0.1 <= next_area[0][3] <= 0.3:
                ncid = 3
            elif 0.1 <= next_area[0][4] <= 0.3:
                ncid = 4
            elif 0.1 <= next_area[0][2] <= 0.3:
                ncid = 2
            elif 0.1 <= next_area[0][5] <= 0.3:
                ncid = 5
            elif 0.1 <= next_area[0][1] <= 0.3:
                ncid = 1
            elif 0.1 <= next_area[0][0] <= 0.3:
                ncid = 0
            second_distance = math.sqrt( (next_area[1][ncid][0] - next_area[2][0])**2 + (next_area[1][cid][1] - next_area[2][1])**2 )

            reward += (first_distance - second_distance) / 2
            # print(first_distance, second_distance, reward)

        if pro_action_num == action_num:
            reward -= 1

        # print(astate[action_num], reward)
        # reward = round(float(int(gamescore) / 60), 2)

        # reward = 1.0

        # if pro_action_num == action_num:
        #     reward = -0.4
        # if STORAGE_TARGET >= reward:
        #     reward = -round(float(math.sqrt(STORAGE_TARGET - reward)), 2)

        return reward


    def save(self):
        print("[-] Model saving...")
        torch.save(self.DQN_eval.state_dict(), MODEL_PATH + str("DQN_eval.pth"))
        torch.save(self.DQN_target.state_dict(), MODEL_PATH + str("DQN_target.pth"))
        print("[+] Model finish save!")

    def save_excellent(self, label=None):
        label = str(label) + "_" + str(time.strftime("%Y%m%d%H%M%S", time.localtime())) + "_excellent"
        print("[-] Model " + str(label) + "saving...")
        torch.save(self.DQN_eval.state_dict(), MODEL_PATH + str(str(label) + "DQN_eval.pth"))
        torch.save(self.DQN_target.state_dict(), MODEL_PATH + str(str(label) + "DQN_target.pth"))
        print("[-] Model " + str(label) + "finish save!")

    def load(self):
        if os.path.exists(MODEL_PATH + str("DQN_eval.pth")) and os.path.exists(MODEL_PATH + str("DQN_target.pth")):
            self.DQN_eval.load_state_dict(torch.load(MODEL_PATH + str("DQN_eval.pth")))
            self.DQN_target.load_state_dict(torch.load(MODEL_PATH + str("DQN_target.pth")))
            print("[+] model load finish!")


if __name__ == '__main__':
    AbrainModel()