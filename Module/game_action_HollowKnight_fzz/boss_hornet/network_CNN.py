from dependencies import *


class DQN(nn.Module):
    def __init__(self, width, height, action):
        super(DQN, self).__init__()
        self.resize_dim = width * height
        self.resize_width = width
        self.resize_height = height
        self.mls = nn.MSELoss()

        # conv2d layer 图片卷积层
        self.conv = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=2),
            nn.BatchNorm2d(8),
            # nn.ReLU(inplace=True),
            nn.ReLU(),

            # nn.MaxPool2d(2, stride=1, ceil_mode=True),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # nn.MaxPool2d(2, stride=2, ceil_mode=True),
            # nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),

        )
        self.dropout = nn.Dropout

        # linear layer 线性层
        # 线性层的输入取决于conv2d的输出，计算输入图像大小
        self.linear_input_size = 14144

        self.linear = nn.Sequential(
            nn.Linear(self.linear_input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, action),
            nn.ReLU(),
        )

        # optimizer = optim.RMSprop(policy_net.parameters())
        self.opt = torch.optim.Adam(self.parameters(), lr=confhk.LEARN_RATE)

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32).to(DEVICE).requires_grad_(True)
        x = self.conv(x)
        x = x.reshape(self.linear_input_size)
        x = self.linear(x)

        return x

