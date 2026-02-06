import torch.nn as nn
import torch
import numpy as np


class channel_wise_attention(nn.Module):
    def __init__(self,H,W,C,reduce):
        super(channel_wise_attention,self).__init__()
        self.H = H  #1
        self.W = W
        self.C = C  #channel=32
        self.r = reduce  #16
        # fc layer
        self.fc = nn.Sequential(
            nn.Linear(self.C,self.r),
            nn.Tanh(),
            nn.Linear(self.r,self.C)
        )
        # softmax
        self.softmax = nn.Softmax(dim=3)

    def forward(self,x):
        # mean pooling
        x1 = x.permute(0,3,2,1)
        mean = nn.AvgPool2d((1, self.W))
        feature_map = mean(x1).permute(0,1,3,2)
        # FC Layer
        # feature_map
        feature_map_fc = self.fc(feature_map)
        
        # softmax
        v = self.softmax(feature_map_fc)
        #print("v")
        #print(v.shape)
        # channel_wise_attention
        v = v.reshape(-1, self.C)
        #print(v)
        #map_cpu = v.cpu()
        #np.save('./wandt/map.npy', map_cpu.detach().numpy())
        vr = torch.reshape(torch.cat([v]*(self.H*self.W),axis=1),[-1,self.H,self.C,self.W])

        channel_wise_attention_fm = x1 * vr
        # ma = channel_wise_attention_fm.cpu()
        # np.save('./m/ma.npy', ma.detach().numpy())
        return channel_wise_attention_fm,v