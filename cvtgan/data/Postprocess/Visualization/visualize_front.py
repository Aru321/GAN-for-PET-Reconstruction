
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 14:15:03 2022

@author: AruZeng
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
resarr=['Ground Truth','LPET','3D U-NET','TransGAN','CVT-GAN','CVT-GAN_edge']
lambda_res=1.
def print_heatmap1(path,images):#传入列表,原SPET图放在第一个
    n=len(images)
    plt.figure(figsize=(5*(n+2),5))
    s_img=np.array(cv2.imread(path+'/'+images[0]+'.jpg',0),dtype=float)

    for i in range(n):
        if(i==0):
            continue
        img=np.array(cv2.imread(path+'/'+images[i]+'.jpg',0),dtype=float)
        plt.subplot(1, n, i+1)
        res=abs(s_img-img)
        #
        sns.heatmap(data=res,cmap=plt.get_cmap('Blues'),vmax=60,xticklabels=False,yticklabels=False)
def print_heatmap2(index,paths):#传入列表,原SPET图放在第一个
    n=len(paths)
    plt.figure(figsize=(5*(n+3),5))
    s_img=np.array(cv2.imread(paths[0]+str(index)+'.jpg',0),dtype=float)
    for i in range(n):
        if(i==0):
            continue
        img=np.array(cv2.imread(paths[i]+str(index)+'.jpg',0),dtype=float)
        plt.subplot(1, n, i+1)
        plt.title(resarr[i],fontsize=20)
        lambda_res=1.
        if(i==1):
            lambda_res=1.5*lambda_res
        res=lambda_res*abs(s_img-img)
        #
        cbar_flag=False
        if(i==n-1):
            cbar_flag=True
        sns.heatmap(data=res[:100,:],cmap=plt.get_cmap('Blues'),vmax=55,xticklabels=False,yticklabels=False)
if __name__ == '__main__':
    for i in range(60,80):
        print_heatmap2(i,['./imgs_front/hp033-pet-720-reorient-align-resample.img/',
                           './imgs_front/hp033-pet-180-1-reorient-align-resample.img/',
                           './imgs_front/unetfake_hp033.img/',
                           './imgs_front/transganfake_hp033.img/',
                            './imgs_front/cut_1000.img/'])
  