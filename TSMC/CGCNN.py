import numpy as np
import torch
import torch.nn as nn
from channel_wise_attention import channel_wise_attention
from Temporal import Temporal


class CGCNN(nn.Module):
    def __init__(self, seq_len, in_chan):
        super(CGCNN, self).__init__()
        self.H = 1
        self.W = seq_len
        self.C = in_chan
        self.reduce = 16
        self.channel_wise_attention = channel_wise_attention(self.H,self.W,self.C,self.reduce)
        self.seq_len = seq_len
        self.Temporal = Temporal(self.seq_len)
        # self.hidden_dim = 16
        # self.lstm = LSTM(self.hidden_dim)
        # self.hidden = 16
        # self.att = self_attention(self.hidden_dim, self.hidden)
        # self.fc = nn.Sequential(
        #     nn.Linear(self.hidden, 4),
        #     nn.Softmax(dim=1)
        # )


    def forward(self,x):
        x_c,v = self.channel_wise_attention(x)
        # x_c = x.permute(0,3,2,1)
        ot, of= self.Temporal(x_c)
        # x_t= self.Temporal(x_c)
        # x_r = self.lstm(x_t)
        # # print(x_r.shape)
        # x_a = self.att(x_r)
        # print(x_a.shape)
        # o = self.fc(x_t)
        return ot, of,v
        # return x_t