
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
        ind=np.where(image_l<2000.0)
        image_l[ind]=0.0
        image_l=np.moveaxis(image_l, [0, 1, 2], [2,1,0])
        savImg = sitk.GetImageFromArray(image_l)
        sitk.WriteImage(savImg,'./data/datastage2/train_ldata'+'/'+file)
                    
    for file in all_s_name:
        image_path_s = os.path.join(root_s,file)
        image_s,h=load(image_path_s)
        image_s=np.array(image_s)
        ind=np.where(image_s<2000.0)
        image_s[ind]=0.0
        image_s=np.moveaxis(image_s, [0, 1, 2], [2,1,0])
        savImg = sitk.GetImageFromArray(image_s)
        sitk.WriteImage(savImg,'./data/datastage2/train_sdata'+'/'+file)
                    
if __name__ == '__main__':
    Datamake('./data/datastage2/train_ldata','./data/datastage2/train_sdata')