import torch
import sys
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from GCU import GCU
from channel_wise_attention import channel_wise_attention


current_module = sys.modules[__name__]



class Temporal(nn.Module):
    def conv_block(self, in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            GCU(in_chan, out_chan,
                      kernel, step),
            #nn.Dropout(0.5),
            # nn.BatchNorm2d(16),
            # nn.Dropout2d(0.3),
            # nn.ReLU(),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))

    def conv_block2(self, in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            GCU(in_chan, out_chan,
                kernel, step),
            # nn.BatchNorm2d(16),
            # nn.Dropout2d(0.3),
            # nn.Dropout(0.5),
            # nn.ReLU(),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))


    def conv_block1(self, in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=step),
            # nn.BatchNorm2d(16),
            # nn.Dropout2d(0.3),
            #nn.LeakyReLU(),
            # nn.ReLU(),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))

    def __init__(self, seq_len):
        # input_size: 1 x EEG channel x datapoint
        super(Temporal, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125, 0.0625]
        self.inception_windw = [1/4, 1/8, 1/16]
        self.pool = 6
        # self.pool1 = 9
        # self.pool2 = 11
        # self.pool3 = 13
        # self.pool4 = 15
        self.num_T0 = 16
        self.num_T = 16
        self.num_T1 =16
        self.num_classes = 4
        self.hidden = 32
        self.fact = 6
        self.fact1 = 2
        self.seq_len = seq_len
        # mlayer = 'LogVarLayer'
        # self.ml = current_module.__dict__[mlayer](dim=2)
        # self.encoder1 = Transformer()
        # self.encoder2 = Transformer1()
        # self.att1 = self_attention(self.num_T, 32)
        # self.att2 = self_attention(self.num_T, 32)
        # self.att3 = self_attention(self.num_T, 32)
        # self.encoder2 = TransformerEncoder(self.d_model2, self.num_heads, self.num_layers, self.drop)
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Temporal1 = self.conv_block2(1, self.num_T1, (1, 2), 1, 1)
        # self.Temporal2 = self.conv_block(1, self.num_T, (1, int(self.inception_windw[1] * self.seq_len)), 1, self.pool)
        # self.Temporal3 = self.conv_block(1, self.num_T, (1, int(self.inception_windw[2] * self.seq_len)), 1, self.pool)
        # self.Temporal4 = self.conv_block(1, self.num_T, (1, int(self.inception_windw[3] * self.seq_len)), 1, self.pool)

        # self.Temporal1 = self.conv_block(1, self.num_T, (1, int(self.inception_windw[0] * self.seq_len)), 1, self.pool1)
        # self.Temporal2 = self.conv_block(1, self.num_T, (1, int(self.inception_windw[1] * self.seq_len)), 1, self.pool2)
        # self.Temporal3 = self.conv_block(1, self.num_T, (1, int(self.inception_windw[2] * self.seq_len)), 1, self.pool3)
        # self.Temporal4 = self.conv_block(1, self.num_T, (1, int(self.inception_windw[3] * self.seq_len)), 1, self.pool4)

        # self.Temporal1 = self.conv_f(1, self.num_T, (1, int(self.inception_windw[0] * self.seq_len)), 1, self.pool)
        # self.Temporal2 = self.conv_f(1, self.num_T, (1, int(self.inception_windw[1] * self.seq_len)), 1, self.pool)
        # self.Temporal3 = self.conv_f(1, self.num_T, (1, int(self.inception_windw[2] * self.seq_len)), 1, self.pool)
        # self.Temporal4 = self.conv_f(1, self.num_T, (1, int(self.inception_windw[3] * self.seq_len)), 1, self.pool)

        # self.Temporal1 = self.conv_f(1, self.num_T, (1, int(self.inception_windw[0] * self.seq_len)), 1, self.pool1)
        # self.Temporal2 = self.conv_f(1, self.num_T, (1, int(self.inception_windw[1] * self.seq_len)), 1, self.pool2)
        # self.Temporal3 = self.conv_f(1, self.num_T, (1, int(self.inception_windw[2] * self.seq_len)), 1, self.pool3)
        # self.Temporal4 = self.conv_f(1, self.num_T, (1, int(self.inception_windw[3] * self.seq_len)), 1, self.pool4)

        self.Temporal5 = self.conv_block(1, self.num_T0, (1, int(self.inception_window[0] * self.seq_len)), 1, self.pool)
        self.Temporal6 = self.conv_block(1, self.num_T0, (1, int(self.inception_window[1] * self.seq_len)), 1, self.pool)
        self.Temporal7 = self.conv_block(1, self.num_T0, (1, int(self.inception_window[2] * self.seq_len)), 1, self.pool)
        self.Temporal8 = self.conv_block(1, self.num_T0, (1, int(self.inception_window[3] * self.seq_len)), 1, self.pool)
        # self.conv1 = self.conv_f(1, self.num_T, (1, 9), (1, 9))
        # self.conv2 = self.conv_f(1, self.num_T, (1, 11), (1, 11))
        # self.conv3 = self.conv_f(1, self.num_T, (1, 13), (1, 13))
        # self.conv4 = self.conv_f(1, self.num_T, (1, 15), (1, 15))

        self.fusion_layer = self.conv_block1(self.num_T, self.num_T, (32, 1), 1, 2)
        self.ff = self.conv_block1(self.num_T1, self.num_T1, (32, 1), 1, 1)
        self.ft = self.conv_block1(self.num_T, self.num_T, (32, 1), 1, 1)

        self.BN_t = nn.BatchNorm2d(self.num_T)
        self.BN_f = nn.BatchNorm2d(self.num_T1)
        self.BN_tf = nn.BatchNorm2d(self.num_T)

        self.BN_fusion = nn.BatchNorm2d(self.num_T)
        self.BN_ff = nn.BatchNorm2d(self.num_T1)
        self.BN_ft = nn.BatchNorm2d(self.num_T)
        self.cv =nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=(1, 14), stride=1)

        self.fcf = nn.Sequential(
            nn.Linear((self.num_T1), self.hidden),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(self.hidden, self.num_classes)
        )

        self.fct = nn.Sequential(
            nn.Linear((self.num_T), self.hidden),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(self.hidden, self.num_classes)
        )
        self.fctf = nn.Sequential(
            nn.Linear(16*106, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4)
        )
        self.f = nn.Linear(192, 4)


    def forward(self, x):
        #############################################
        ##########################################################################
        fft = torch.rfft(x, signal_ndim=1,normalized=True,onesided=True)
        # # # print(fft.shape)
        fft = fft[:,:,:,1:60]
        f =torch.sqrt(fft[...,0]**2+fft[...,1]**2)
        # print(f.shape)
        # # # # x = fft[...,0]
        of = self.Temporal1(f)
        # fu = of.cpu()
        # np.save('./m/fu.npy', fu.detach().numpy())
        # print(of.shape)
        of1 = self.BN_f(of)

        ###########################################################################
        #print(of.shape)
        ###########################################################################
        y1 = self.Temporal5(x)
        # print(y1.shape)
        y2 = self.Temporal6(x)
        # print(y2.shape)
        y3 = self.Temporal7(x)
        # y4 = self.Temporal8(x)
        #print(y3.shape)
        ot = torch.cat((y1, y2, y3), dim=-1)
        tu = ot.cpu()
        np.save('./m/tu.npy', tu.detach().numpy())
        # ot = torch.cat((y1, y2, y3,y4), dim=-1)
        ot1 = self.BN_t(ot)
        #print(ot1.shape)
        ot1 = torch.cat((of1,ot1),dim=-1)

        ot = self.ft(ot1)
        ot = self.BN_ft(ot)
        #print(ot.shape)
        ot1 = torch.squeeze(torch.mean(ot, dim=-1), dim=-1)
        # ft = ot1.cpu()
        # np.save('./m/ft.npy', ft.detach().numpy())
        #  print(ot1.shape)
        # ot = self.att2(ot)
        ot = self.fct(ot1)
        # ot = F.softmax(ot,dim=-1)

        ####################################################################


        return ot, ot1