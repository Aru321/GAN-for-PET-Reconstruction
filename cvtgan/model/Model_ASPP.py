# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 11:04:45 2022

@author: AruZeng
"""
from torch import nn
import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
class SimpleAttention(nn.Module):
     def __init__(self, in_channels, out_channels):
         super().__init__()
         self.act=nn.LeakyReLU()
         self.emb=nn.Conv3d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1)
         self.encoder_layer = nn.TransformerEncoderLayer(d_model=out_channels, nhead=4)
     def forward(self,x):
         x=self.emb(x)
         x=self.act(x)
         size=x.shape
         x=rearrange(x, 'b c h w d  -> b (h w d) c')
         x=self.encoder_layer(x)
         out=rearrange(x, 'b (h w d) c  -> b c h w d',h=size[-3],w=size[-2],d=size[-1])
         return out
     
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv3d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())
    def forward(self, x):
        size = x.shape[-3:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='trilinear', align_corners=False)
class ASPP_with_attn(nn.Module):
    def __init__(self, in_channels,out_channels, atrous_rates):
        super(ASPP_with_attn, self).__init__()
        out_channels = out_channels
        modules = []
        modules.append(nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()))
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(SimpleAttention(in_channels, out_channels))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv3d(6 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
#
class ASPP(nn.Module):
    def __init__(self, in_channels,out_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = out_channels
        modules = []
        modules.append(nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()))
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv3d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
#
class DownSamp(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DownSamp,self).__init__()
        self.block1=nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3,stride=2,padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU())
    def forward(self,x):
        x=self.block1(x)
        return x
class ResBlock3D(nn.Module):
    def __init__(self,in_chan,out_chan):
        super().__init__()
        self.in_chan=in_chan
        self.out_chan=out_chan
        self.conv=nn.Conv3d(in_chan,out_chan,kernel_size=1,stride=1,padding=0)
    def forward(self,x):
        return self.conv(x)
    
class ASPPNet3D(nn.Module):
    def __init__(self,in_channels):
        super(ASPPNet3D,self).__init__()
        self.down=DownSamp(in_channels,64)
        in_channels=64
        self.block1=ASPP(in_channels,2*in_channels,[3,6,9])
        self.res1=ResBlock3D(in_channels, 2*in_channels)
        #.block2=ASPP(2*in_channels,4*in_channels,[2,4,6])
        #self.block3=ASPP(4*in_channels,2*in_channels,[2,4,6])
        self.block4=ASPP(2*in_channels,in_channels,[3,6,9])
        self.res2=ResBlock3D(2*in_channels, in_channels)
        self.out=ASPP(in_channels,1,[4,8,12])
    def forward(self,x):
        inx=x
        size=x.shape[-3:]
        x=self.down(x)
        x=self.block1(x)+self.res1(x)
        #x=F.interpolate(x, size=size, mode='trilinear', align_corners=False)
        #x=self.block2(x)
        #x=self.block3(x)
        x=self.block4(x)+self.res2(x)
        x=F.interpolate(x, size=size, mode='trilinear', align_corners=False)
        x=self.out(x)+inx
        return x
def G_train_L1(G:ASPPNet3D, X, Y, L1, optimizer_G, device,lamb=100):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #image_size = X.size(3) // 2
    x = X.to(device)   
    y = Y.to(device)   
    #x = X[:, :, :, image_size:]   
    #y = X[:, :, :, :image_size]   
    G.zero_grad()
    # fake data
    G_output = G(x)
    X_fake = torch.cat([x, G_output], dim=1)
    #print(D_output_f.shape)
    #G_BCE_loss = BCELoss(D_output_f, torch.ones(D_output_f.size()))
    G_L1_Loss = L1(G_output, y)
    #print('G_L1_loss:',G_L1_Loss.item())
    print('L1_loss:',G_L1_Loss.data.item())
    #
    G_loss = lamb * G_L1_Loss
    #print('cur_g_loss:',G_loss.item())
    G_loss.backward()
    optimizer_G.step()
    return G_loss.data.item()       
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    aspp = ASPPNet3D(1).to(device)
    total_params = sum(p.numel() for p in aspp.parameters())
    print(total_params)
    aspp.eval()
    x=torch.randn(1,1,64,64,64).to(device)
    x=aspp(x)
    print(x.shape)
     