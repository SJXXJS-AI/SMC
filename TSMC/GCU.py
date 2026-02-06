import torch
import torch.nn as nn

class GCU(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, step):
        super(GCU, self).__init__()

        # 定义两个卷积层，分别用于计算特征和门控信号
        self.conv_f = nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=kernel, stride=step)
        self.conv_g = nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=kernel, stride=step)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算特征和门控信号
        f = self.conv_f(x)
        g = self.sigmoid(self.conv_g(x))

        # 将特征和门控信号相乘，得到加入门控单元后的结果
        out = torch.mul(f, g)

        return out