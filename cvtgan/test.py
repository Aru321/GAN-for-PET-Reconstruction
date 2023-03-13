# -*- coding: utf-8 -*-
import medpy
from medpy.io import load
import data.Preprocess.datautils3d as util3d
import model.CVT3D_Main
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
from numpy import mean,square
#
from skimage.metrics import normalized_root_mse
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import math
from math import sqrt
#
from compute_dr import compute_dr 
def normalize(tests):
    image_s=(tests-tests.min())/(tests.max()-tests.min())
    return image_s
sitk.ProcessObject.SetGlobalWarningDisplay(False)
#导入数据
testloader=util3d.loadData('./data/predict_results','','./data/testmci_sdata','',batch_size=1,shuffle=False)
#模型导入0
SSIM=0
PSNR=0
NMSE=0
cnt=1.
#
clips=[]
clips_s=[]
for res, image_s in testloader:
    #
    image_s=np.squeeze(image_s.detach().numpy())
    image_s=normalize(image_s)
    res=res.cpu().detach().numpy()
    #res=normalize(res)
    res=np.squeeze(res)
    #print(res.shape,image_s.shape)
    #
    y=np.nonzero(image_s)#取非黑色部分
    image_s_1=image_s[y]
    print(image_s_1.shape)
    #image_l_1=image_l[y]
    res_1=res[y]
    dr=compute_dr(image_s,res)
    #计算PSNR
    cur_psnr=(peak_signal_noise_ratio(res_1,image_s_1,data_range=1))
    PSNR=(PSNR+cur_psnr)
    print("Cur_PSNR:",cur_psnr,"total_psnr:",PSNR/cnt)
    cur_ssim=structural_similarity(res,image_s,multichannel=True,data_range=1)
    SSIM=(SSIM+cur_ssim)
    print(" Cur_SSIM:",cur_ssim,"total_ssim:",SSIM/cnt)
    NMSE+=(normalized_root_mse(image_s, res))**2
    #计算SSIM
    #
    '''
    #计算PSNR
    PSNR=peak_signal_noise_ratio(image_l_1,image_s_1,data_range=dr_1)
    #计算SSIM
    SSIM=structural_similarity(image_l_1,image_s_1,data_range=dr_1)
    '''
    #NMSE+=(normalized_root_mse(image_s_1, res_1))**2
    #
    print("Cur_NMSE:",NMSE/cnt)
    cnt+=1

 