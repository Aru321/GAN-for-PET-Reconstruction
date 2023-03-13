# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 13:21:05 2021

@author: AruZeng
"""
import medpy
from medpy.io import load
from medpy.io import save
import numpy as np
import os
import SimpleITK as sitk
#用于数据切片
def Datamake(root_l,root_s):
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
       # print(cut_cnt)
        for i in range(0,8):
            for j in range(0,8):
                for k in range(0,8):
                    image_cut=image_l[9*i:64+9*i,9*j:64+9*j,9*k:64+9*k]
                    savImg = sitk.GetImageFromArray(image_cut)
                    sitk.WriteImage(savImg,'./dataset/processed/LPET_cut'+'/'+file+'_cut'+str(cut_cnt)+'.img')
                    cut_cnt+=1
                    
    for file in all_s_name:
        image_path_s = os.path.join(root_s,file)
        image_s,h=load(image_path_s)
        image_s=np.array(image_s)
        #print(image_l.shape)
        cut_cnt=0
        for i in range(0,8):
            for j in range(0,8):
                for k in range(0,8):
                    image_cut=image_s[9*i:64+9*i,9*j:64+9*j,9*k:64+9*k]
                    savImg = sitk.GetImageFromArray(image_cut)
                    sitk.WriteImage(savImg,'./dataset/processed/SPET_cut'+'/'+file+'_cut'+str(cut_cnt)+'.img')
                    cut_cnt+=1
#
def Datamake_phantom(root_l, root_s):
    all_l_names = []
    all_s_names = []
    for root, dirs, files in os.walk(root_l):
        all_l_names = (files)
    for root, dirs, files in os.walk(root_s):
        all_s_names = (files)
    #
    all_l_name = []
    all_s_name = []
    for i in all_l_names:
        if os.path.splitext(i)[1] == ".img":
            # print(i)
            all_l_name.append(i)
    for i in all_s_names:
        if os.path.splitext(i)[1] == ".img":
            all_s_name.append(i)
    #
    print(all_l_name)
    #
    for file in all_l_name:
        image_path_l = os.path.join(root_l, file)
        image_l, h = load(image_path_l)
        image_l = np.array(image_l)
        # print(image_l.shape)
        cut_cnt = 0
        # print(cut_cnt)
        for i in range(0, 8):
            for j in range(0, 8):
                for k in range(0, 8):
                    image_cut = image_l[9 * i:64 + 9 * i, 9 * j:64 + 9 * j, 9 * k:64 + 9 * k]
                    savImg = sitk.GetImageFromArray(image_cut)
                    sitk.WriteImage(savImg, './dataset/processed/Lphant_cut' + '/' + file + '_cut' + str(cut_cnt) + '.img')
                    cut_cnt += 1

    for file in all_s_name:
        image_path_s = os.path.join(root_s, file)
        image_s, h = load(image_path_s)
        image_s = np.array(image_s)
        # print(image_l.shape)
        cut_cnt = 0
        for i in range(0, 8):
            for j in range(0, 8):
                for k in range(0, 8):
                    image_cut = image_s[9 * i:64 + 9 * i, 9 * j:64 + 9 * j, 9 * k:64 + 9 * k]
                    savImg = sitk.GetImageFromArray(image_cut)
                    sitk.WriteImage(savImg, './dataset/processed/Sphant_cut' + '/' + file + '_cut' + str(cut_cnt) + '.img')
                    cut_cnt += 1
#
def DatamakeMulti(root_mri):
    all_mri_names=[]
    for root, dirs, files in os.walk(root_mri):
        all_mri_names=(files)
    all_mri_name=[]
    for i in all_mri_names:
        if os.path.splitext(i)[1] == ".img":
            all_mri_name.append(i)
    #
    print(all_mri_name)
    for file in all_mri_name:
        image_path_mri = os.path.join(root_mri,file)
        image_mri,h=load(image_path_mri)
        image_mri=np.array(image_mri)
        #print(image_l.shape)
        cut_cnt=0
        for i in range(0,8):
            for j in range(0,8):
                for k in range(0,8):
                    image_cut=image_mri[9*i:64+9*i,9*j:64+9*j,9*k:64+9*k]
                    savImg = sitk.GetImageFromArray(image_cut)
                    sitk.WriteImage(savImg,'./dataset/processed/T1_cut'+'/'+file+'_cut'+str(cut_cnt)+'.img')
                    cut_cnt+=1
                    
if __name__ == '__main__':
    Datamake_phantom('./dataset/processed/Lphant','./dataset/processed/Sphant')
    #DatamakeMulti('./dataset/processed/T1')
 