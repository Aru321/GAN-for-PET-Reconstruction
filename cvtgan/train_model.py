# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 11:35:23 2021

@author: AruZeng
"""
#
import data.Preprocess.datautils3d as util3d
import model.CVT3D_Model as CVT3D_Model
#
import SimpleITK as sitk
import pickle
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import pandas as pd
import os
import cv2
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import argparse
# 1. 超参数argparser
# 2. 验证集切分：读取前缀的文件，Dataset文件重写
# 3. 没几个epoch用验证集测试指标，临时文件夹
# 4. 模型保存： the best

def featureScaling(arr):
    scaler = MinMaxScaler()
    result = scaler.fit_transform(arr)
    return result
'''
for data,label in trainloader:
    print('data:',data.shape)
    print('label:',data.shape)
'''
# file_txt_dir file_dir_str
# devices device_str
# lr_G float
# lr_D float
# lamb float
# epochs int
# batch_size int



# beta1 float
# resume - checkpoint boolean
# use_checkpoint boolean?
# checkpoint_dir file_dir_str
# val_epoch_inv int
# temp_val_img_cut file_dir_str
# res_model file_dir_str
# pretrained_model boolean
# pretrained_model_dir file_dir_str

def parse_option():
    parser = argparse.ArgumentParser('CVT3D training and evaluation script', add_help=False)
    # easy config modification
    parser.add_argument('--file_txt_dir', type=str, help='path split txt file to dataset')
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--devices', default='0', type=str,
                        help='Set the CUDA_VISIBLE_DEVICES env var from this string')
    parser.add_argument('--lr_G', '--learning-rate-G', default=2e-4, type=float, help='initial learning rate of G',
                        dest='lr_G')
    parser.add_argument('--lr_D', '--learning-rate-D', default=2e-4, type=float, help='initial learning rate of D',
                        dest='lr_D')
    parser.add_argument('--lamb', '--lambda', default=100, type=float, dest='lamb')
    parser.add_argument('--beta1', default=0.9, type=float, dest='beta1')
    parser.add_argument('--pretrained', type=bool, help='pretrained weight')
    parser.add_argument('--pretrained_model_dir', type=str, help='pretrained weight path')
    parser.add_argument('--resume', type=bool, help='resume from checkpoint')
    parser.add_argument('--use-checkpoint', action='store_true',help="")
    parser.add_argument('--checkpoint_dir', type=str, help='pretrained weight path')

    parser.add_argument('--val_epoch_inv', default=5, type=int, help='validation interval epochs')
    parser.add_argument('--temp_val_img_cut', type=str, help='pretrained weight path')


    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def main():
    #数据集
    trainloader=util3d.loadData('./data/train_l_cut','','./data/train_s_cut','',batch_size=1,shuffle=False)
    #device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    print("len:",len(trainloader))
    #params
    lr_G, lr_D = 0.0002, 0.0002  # 学习率
    beta1 = 0.9  # 动量
    lamb = 100  # L1
    epochs = 100  
    #模型
    #pretrained_g=torch.load('generator.pkl')
    #pretrained_d=torch.load('discriminator.pkl')
    G = CVT3D_Model.Generator().to(device)
    #G=pretrained_g.to(device)
    D = CVT3D_Model.Discriminator().to(device)
    #D=pretrained_d.to(device)
    #模型大小查看
    total_params = sum(p.numel() for p in G.parameters())
    print(total_params)
    #
    # 目标函数 & 优化器
    BCELoss = nn.BCELoss().to(device)
    #BCELoss = nn.BCELoss()
    L1 = nn.L1Loss().to(device)  
    #L1 = nn.L1Loss()
    #首先进行第一阶段训练
    #
    #
    '''
    optimizer_G_s1 = optim.Adam(G.parameters(), lr=0.001, betas=(beta1, 0.999))
    #从断点开始训练
    start_epoch = 0
    resume = False
    if resume:
        if os.path.isfile('checkpoint_stage1'):
           checkpoint = torch.load('checkpoint_stage1')
           start_epoch = checkpoint['epoch'] + 1
           G.load_state_dict(checkpoint['G'])
           optimizer_G_s1.load_state_dict(checkpoint['optimizer_G'])
           print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
           print("=> no checkpoint found")
    #
    G_Loss=[]
    for epoch in range(start_epoch,10):
        G_losses, batch, g_l = [],0,0
        for X, Y in trainloader:
            #归一化
            X=(X-X.min())/(X.max()-X.min())
            Y=(Y-Y.min())/(Y.max()-Y.min())
            #
            batch += 1
            if batch==1:
                print(X)
                print(Y)
            #
    
            #
            G_losses.append(CVT3D.G_train_stage_one(G,X,Y,L1, optimizer_G_s1,device))
            g_l = np.array(G_losses).mean()
            print('[%d / %d]: batch#%d loss_g= %.3f' %
                  (epoch + 1, epochs, batch, g_l))
        G_Loss.append(g_l)
    #
        checkpoint_stage1 = {
                'epoch': epoch,
                'G': G.state_dict(),
                'optimizer_G': optimizer_G_s1.state_dict()
            }
        torch.save(checkpoint_stage1,'checkpoint_stage1')
    #
    torch.save(G, 'generator_stage1.pkl')
    '''
    #第二阶段训练-对抗训练
    #G=torch.load('generator_stage1.pkl')
    #checkpoint = torch.load('checkpoint_stage1')
    #G.load_state_dict(checkpoint['G'])
    #
    #G=torch.load('generator_with_edge.pkl').to(device)
    #
    optimizer_G = optim.Adam(G.parameters(), lr=lr_G, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=lr_D, betas=(beta1, 0.999))
    #从断点开始训练
    start_epoch = 0
    resume = False
    if resume:
        if os.path.isfile('checkpoint'):
           checkpoint = torch.load('checkpoint')
           start_epoch = checkpoint['epoch'] + 1
           G.load_state_dict(checkpoint['G'])
           D.load_state_dict(checkpoint['D'])
           optimizer_G.load_state_dict(checkpoint['optimizer_G'])
           optimizer_D.load_state_dict(checkpoint['optimizer_D'])
           print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
           print("=> no checkpoint found")
    #
    #训练
    D_Loss, G_Loss, Epochs = [], [], range(1, epochs + 1)  # 一次epoch的loss
    ERROR_CNT=0
    torch.cuda.empty_cache()
    for epoch in range(start_epoch,epochs):
        D_losses, G_losses, batch, d_l, g_l = [], [], 0, 0, 0  
        for x, y in trainloader:
            batch += 1
             #归一化
            X=(x-x.min())/(x.max()-x.min())
            Y=(y-y.min())/(y.max()-y.min())
            # 训练Discriminator
            d_loss=CVT3D_Model.D_train(D, G, X,Y,BCELoss, optimizer_D,device)
            D_losses.append(d_loss)
            # 训练Generator
            g_loss=CVT3D_Model.G_train(D, G, X,Y,BCELoss, L1, optimizer_G,device,lamb)
            '''
            if g_loss>100:
                savImgX = sitk.GetImageFromArray(X)
                sitk.WriteImage(savImgX,'./error'+'/'+'errorx_'+str(ERROR_CNT)+'.img')
                savImgY = sitk.GetImageFromArray(Y)
                sitk.WriteImage(savImgY,'./error'+'/'+'errory_'+str(ERROR_CNT)+'.img')
                ERROR_CNT+=1
            '''
            if g_loss>0:
                G_losses.append(g_loss)
            #平均loss
            d_l, g_l = np.array(D_losses).mean(), np.array(G_losses).mean()
            print('[%d / %d]: batch#%d loss_d= %.3f  loss_g= %.3f' %
                  (epoch + 1, epochs, batch, d_l, g_l))
        # 保存每次epoch的loss
        D_Loss.append(d_l)
        G_Loss.append(g_l)
        #
        if(epoch%5==0):
            torch.save(G, 'generator'+'epoch_'+str(epoch)+'.pkl')
        # 保存训练储存点
        checkpoint = {
            'epoch': epoch,
            'G': G.state_dict(),
            'D': D.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D': optimizer_D.state_dict()
        }
        torch.save(checkpoint,'checkpoint')
        #
        
    #保存模型
    torch.save(G, 'generator.pkl')
    torch.save(D, 'discriminator.pkl')
## 运行 ##
if __name__ == '__main__':
    main()
    
    
