# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 15:01:11 2022

@author: AruZeng
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 16:49:38 2021

@author: AruZeng
"""
import medpy
from medpy.io import load
import data.Preprocess.datautils3d as util3d
import model.CVT3D_Main,model.CVT3D_Model
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
import math
from math import sqrt  
stride_3d = [32,32,32]
window_3d = [64,64,64]
#导入数据
testloader=util3d.loadData('./data/testmci_l_cut','','./data/testmci_s_cut','',batch_size=1,shuffle=False)
#模型导入0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
G=torch.load('./trained_models/generator_small.pkl').to(device)
SSIM=0
PSNR=0
NMSE=0
cnt=0.
#
clips=[]
clips_s=[]
clips_l=[]
G.eval()
for image_l, image_s in testloader:
    cnt+=1.0
    #归一化
    testl=image_l
    l_min=testl.min()
    l_max=testl.max()
    image_l=(testl-testl.min())/(testl.max()-testl.min())
    image_l=image_l.to(device)#
    #
    tests=image_s
    s_min=tests.min()
    s_max=tests.max()
    image_s=(tests-tests.min())/(tests.max()-tests.min())
    image_s=np.squeeze(image_s.detach().numpy())
    #ind=np.where(image_s<50)
    #image_s[ind]=1
    #预测结果
    # infer_size = window_size
    '''
    B,C,D,H,W = image_l.shape
    s_d,s_h,s_w = stride_3d
    w_d,w_h,w_w = window_3d
    counter = torch.zeros_like(image_l[0])
    num_d = (D - w_d)//s_d + 1
    num_h = (H - w_h)//s_h + 1
    num_w = (W - w_w)//s_w + 1
    res_collect = torch.zeros_like(image_l)
    print(num_d,num_h,num_w)
    for i in range(num_d):
        for j in range(num_h):
            for k in range(num_w):
                counter[i*s_d:i*s_d+w_d, j*s_h:j*s_h+w_h, k*s_w:k*s_w+w_w] += 1
                x = image_l[:,:,i*s_d:i*s_d+w_d, j*s_h:j*s_h+w_h, k*s_w:k*s_w+w_w]
                print(x.shape)
                y = G(x)
                res_collect[:,:,i*s_d:i*s_d+w_d, j*s_h:j*s_h+w_h, k*s_w:k*s_w+w_w] += y
                del y,x
                torch.cuda.empty_cache()
    res_collect /= counter
    res=res_collect
    '''
    res=G(image_l)
    #res=res*(l_max-l_min)+l_min
    res=res.cpu().detach().numpy()
    res=np.squeeze(res)
    #
    image_l=image_l.cpu().detach().numpy()
    image_l=np.squeeze(image_l)
    #
    savImg = sitk.GetImageFromArray(res)
    cnt1=int(cnt)
    filename=f'cut_{cnt1:04d}'+'.img'
    sitk.WriteImage(savImg,'./predicted_nc_patches'+'/'+filename)
    
 