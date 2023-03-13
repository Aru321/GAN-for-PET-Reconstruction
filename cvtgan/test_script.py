
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 14:06:24 2021

@author: AruZeng
"""
import shutil
import medpy
from medpy.io import load
import datautils3d as util3d
from predict_patch import predict_patch_mci,predict_patch_nc
from concat_pathes import concat
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
from skimage.metrics import normalized_root_mse
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
#
def del_file(filepath):
    """
    删除某一目录下的所有文件或文件夹
    :param filepath: 路径
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
#
def testDatamake(root_l,root_s,pre):
    all_l_names=[]
    all_s_names=[]
    for root, dirs, files in os.walk(root_l):
        all_l_names=(files)
    for root, dirs, files in os.walk(root_s):
        all_s_names=(files)
    #
    all_l_name_0=[]
    all_s_name_0=[]
    for i in all_l_names:
        if os.path.splitext(i)[1] == ".img":
            #print(i)
            all_l_name_0.append(i)
    for i in all_s_names:
        if os.path.splitext(i)[1] == ".img":
            all_s_name_0.append(i)
    #
    all_l_name=[]
    all_s_name=[]
    #
    #根据pre选择对应文件
    for file in all_l_name_0:
        for p in pre:
            if p in file:
                all_l_name.append(file)
    for file in all_s_name_0:
        for p in pre:
            if p in file:
                all_s_name.append(file)
    print(all_l_name)
    print(all_s_name)
    #清除所有文件
    del_file('./data/testmci_l_cut/') 
    del_file('./data/testmci_s_cut/') 
    del_file('./data/testnc_l_cut/')
    del_file('./data/testnc_s_cut/') 
    for file in all_l_name:
        image_path_l = os.path.join(root_l,file)
        image_l,h=load(image_path_l)
        image_l=np.array(image_l)
        #print(image_l.shape)
        cut_cnt=0
        for i in range(0,5):
            for j in range(0,5):
                for k in range(0,5):
                    image_cut=image_l[16*i:64+16*i,16*j:64+16*j,16*k:64+16*k]
                    savImg = sitk.GetImageFromArray(image_cut)
                    sitk.WriteImage(savImg,'./data/testnc_l_cut'+'/'+file+'_cut'+f'{cut_cnt:04d}'+'.img')
                    cut_cnt+=1
    for file in all_s_name:
            image_path_s = os.path.join(root_s,file)
            image_s,h=load(image_path_s)
            image_s=np.array(image_s)
            #print(image_l.shape)
            cut_cnt=0
            for i in range(0,5):
                for j in range(0,5):
                    for k in range(0,5):
                        image_cut=image_s[16*i:64+16*i,16*j:64+16*j,16*k:64+16*k]
                        savImg = sitk.GetImageFromArray(image_cut)
                        sitk.WriteImage(savImg,'./data/testnc_s_cut'+'/'+file+'_cut'+f'{cut_cnt:04d}'+'.img')
                        cut_cnt+=1    
#
fix=1.0285
def testDatamake_mci(root_l,root_s,pre):
    all_l_names=[]
    all_s_names=[]
    for root, dirs, files in os.walk(root_l):
        all_l_names=(files)
    for root, dirs, files in os.walk(root_s):
        all_s_names=(files)
    #
    all_l_name_0=[]
    all_s_name_0=[]
    for i in all_l_names:
        if os.path.splitext(i)[1] == ".img":
            #print(i)
            all_l_name_0.append(i)
    for i in all_s_names:
        if os.path.splitext(i)[1] == ".img":
            all_s_name_0.append(i)
    #
    all_l_name=[]
    all_s_name=[]
    #
    #根据pre选择对应文件
    for file in all_l_name_0:
        for p in pre:
            if p in file:
                all_l_name.append(file)
    for file in all_s_name_0:
        for p in pre:
            if p in file:
                all_s_name.append(file)
    #
    del_file('./data/testmci_l_cut/') 
    del_file('./data/testmci_s_cut/') 
    del_file('./data/testnc_l_cut/')
    del_file('./data/testnc_s_cut/') 
    #
    print(all_l_name)
    print(all_s_name)
    for file in all_l_name:
        image_path_l = os.path.join(root_l,file)
        image_l,h=load(image_path_l)
        image_l=np.array(image_l)
        #print(image_l.shape)
        cut_cnt=0
        for i in range(0,5):
            for j in range(0,5):
                for k in range(0,5):
                    image_cut=image_l[16*i:64+16*i,16*j:64+16*j,16*k:64+16*k]
                    savImg = sitk.GetImageFromArray(image_cut)
                    sitk.WriteImage(savImg,'./data/testmci_l_cut'+'/'+file+'_cut'+f'{cut_cnt:04d}'+'.img')
                    cut_cnt+=1
    for file in all_s_name:
            image_path_s = os.path.join(root_s,file)
            image_s,h=load(image_path_s)
            image_s=np.array(image_s)
            #print(image_l.shape)
            cut_cnt=0
            for i in range(0,5):
                for j in range(0,5):
                    for k in range(0,5):
                        image_cut=image_s[16*i:64+16*i,16*j:64+16*j,16*k:64+16*k]
                        savImg = sitk.GetImageFromArray(image_cut)
                        sitk.WriteImage(savImg,'./data/testmci_s_cut'+'/'+file+'_cut'+f'{cut_cnt:04d}'+'.img')
                        cut_cnt+=1 
def normalize(tests):
    image_s=(tests-tests.min())/(tests.max()-tests.min())
    image_s[0,0,0]=fix
    return image_s
def test_results_mci(pre):
    #导入数据
    testloader=util3d.loadData('./data/predict_results/','','./data/testmci_sdata/','',batch_size=1,prefixs=pre,shuffle=False)
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
        if(res.max()>1):
            res=normalize(res)
        #print(res.shape,image_s.shape)
        #
        y=np.nonzero(image_s)#取非黑色部分
        image_s_1=image_s[y]
        print(image_s_1.shape)
        #image_l_1=image_l[y]
        res_1=res[y]
        dr=max(np.max(image_s),np.max(res_1))-min(np.min(image_s),np.min(res_1))
        #计算PSNR
        cur_psnr=(peak_signal_noise_ratio(image_s,res,data_range=1))
        PSNR=(PSNR+cur_psnr)
        print("Cur_PSNR:",cur_psnr,"total_psnr:",PSNR/cnt)
        cur_ssim=structural_similarity(res,image_s,multichannel=True)
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
#
def test_results_nc(pre):
    #导入数据
    testloader=util3d.loadData('./data/predict_results','','./data/testnc_sdata','',batch_size=1,prefixs=pre,shuffle=False)
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
        if(res.max()>1):
            res=normalize(res)
        #print(res.shape,image_s.shape)
        #
        y=np.nonzero(image_s)#取非黑色部分
        image_s_1=image_s[y]
        print(image_s_1.shape)
        #image_l_1=image_l[y]
        res_1=res[y]
        #计算PSNR
        cur_psnr=(peak_signal_noise_ratio(image_s,res,data_range=1.))
        PSNR=(PSNR+cur_psnr)
        print("Cur_PSNR:",cur_psnr,"total_psnr:",PSNR/cnt)
        cur_ssim=structural_similarity(res,image_s,multichannel=True)
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
#
def testflow_mci(pre,model_name):
    testDatamake_mci('./data/testmci_ldata','./data/testmci_sdata',pre)
    predict_patch_mci(model_name,pre)
    concat()
    test_results_mci(pre)
def testflow_nc(pre,model_name):
    testDatamake('./data/testnc_ldata','./data/testnc_sdata',pre)
    predict_patch_nc(model_name,pre)
    concat()
    test_results_nc(pre)
    
if __name__ == '__main__':
    sitk.ProcessObject.SetGlobalWarningDisplay(False)
    #testflow_mci(['MCI_007'],'best_PSNR_model_MCI007_2.pkl')
    #testflow_nc(['hp009'],'best_PSNR_model_hp9.pkl')
    #test_results_nc(['hp'])
    test_results_mci(['MCI'])
    #test_results_mci(['fake'])
    