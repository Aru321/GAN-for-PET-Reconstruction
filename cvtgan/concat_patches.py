# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 11:08:54 2022

@author: AruZeng
"""
import numpy as np
from skimage import io
import SimpleITK as sitk
from medpy.io import load
import scipy.io as sio
from glob import glob
import os
import copy as cp
from numpy.lib import stride_tricks
if __name__ == '__main__':
    all_names=[]
    for root, dirs, files in os.walk('./predicted_nc_patches'):
        all_names=(files)
    #print(all_names)
    all_name=[]
    for i in all_names:
        if os.path.splitext(i)[1] == ".img":
            #print(i)
            all_name.append(i)
    #all_name=all_name.sort(key=lambda k:(int(k[-7:-4])))
    print(all_name)
    cnt=0
    clips=[]
    stride_3d = [16,16,16]
    window_3d = [64,64,64]
    for file in all_name:
        image_path = os.path.join('./predicted_nc_patches',file)
        image,h=load(image_path)
        image=np.array(image)
        #image=np.moveaxis(image, [0, 1, 2], [2,1,0])
        clips.append(image)
        if(len(clips)==125):
            cnt=0
            s_d,s_h,s_w = stride_3d
            w_d,w_h,w_w = window_3d
            counter = np.zeros([128,128,128])
            D,H,W=counter.shape
            num_d = (D - w_d)//s_d + 1
            num_h = (H - w_h)//s_h + 1
            num_w = (W - w_w)//s_w + 1
            res_collect = np.zeros([128,128,128])
            print(num_d,num_h,num_w)
            for i in range(num_d):
                for j in range(num_h):
                    for k in range(num_w):
                        counter[i*s_d:i*s_d+w_d, j*s_h:j*s_h+w_h, k*s_w:k*s_w+w_w] += 1
                        x = clips[cnt]
                        cnt+=1
                        res_collect[i*s_d:i*s_d+w_d, j*s_h:j*s_h+w_h, k*s_w:k*s_w+w_w] += x
            res_collect /= counter
            res=res_collect
            cnt=0
            clips=[]
            res=np.moveaxis(res, [0, 1, 2], [2,1,0])
            y=np.where(res<0.01)
            res[y]=0.0
            savImg = sitk.GetImageFromArray(res)
            sitk.WriteImage(savImg,'./data/predict_results'+'/'+file)