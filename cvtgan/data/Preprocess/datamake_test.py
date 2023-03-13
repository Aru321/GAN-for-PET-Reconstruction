# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 14:06:24 2021

@author: AruZeng
"""
import medpy
from medpy.io import load
import datautils3d as util3d
import CVT3D_Main
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
#
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
def testDatamake(root_l,root_s):
    all_l_names=[]
    all_s_names=[]
    for root, dirs, files in os.walk(root_l):
        all_l_names=(files)
    for root, dirs, files in os.walk(root_s):
        all_s_names=(files)
    #
    all_l_name=[]
    all_s_name=[]
    for i in all_l_names:
        if os.path.splitext(i)[1] == ".img":
            #print(i)
            all_l_name.append(i)
    for i in all_s_names:
        if os.path.splitext(i)[1] == ".img":
            all_s_name.append(i)
    #
    print(all_l_name)
    #
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
def testDatamake_mci(root_l,root_s):
    all_l_names=[]
    all_s_names=[]
    for root, dirs, files in os.walk(root_l):
        all_l_names=(files)
    for root, dirs, files in os.walk(root_s):
        all_s_names=(files)
    #
    all_l_name=[]
    all_s_name=[]
    for i in all_l_names:
        if os.path.splitext(i)[1] == ".img":
            #print(i)
            all_l_name.append(i)
    for i in all_s_names:
        if os.path.splitext(i)[1] == ".img":
            all_s_name.append(i)
    #
    print(all_l_name)
    #
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
if __name__ == '__main__':
    testDatamake('./data/testnc_ldata','./data/testnc_sdata')
    testDatamake_mci('./data/testmci_ldata','./data/testmci_sdata')







