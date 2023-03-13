# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 14:15:03 2022

@author: AruZeng
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
resarr=['Ground Truth','LPET','AutoContext','3D-cGAN','3D-Unet','TransformerGAN','CVT-GAN']
ablation=['Ground Truth','LPET','ab0','ab1','CVT_GAN']
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
def print_heatmap2(index,paths,index_fix):#传入列表,原SPET图放在第一个
    n=len(paths)
    plt.figure(figsize=(5*(n+1),5))
    s_img=np.array(cv2.imread(paths[0]+str(index)+'.jpg',0),dtype=float)
    for i in range(n):
        if(i==0):
            continue
        img=np.array(cv2.imread(paths[i]+str(index-index_fix[i])+'.jpg',0),dtype=float)
        plt.subplot(1, n, i+1)
        plt.title(resarr[i],fontsize=30)
        lambda_res=1.
        if (i==1):
            lambda_res=1.8*lambda_res
        elif (i==2):
            lambda_res=0.6*lambda_res
        res=lambda_res*abs(s_img-img)
        #
        cbar_flag=False
        if(i==n-1):
            cbar_flag=True
        sns.heatmap(data=res,cmap=plt.get_cmap('Blues'),vmax=55,xticklabels=False,yticklabels=False)
def print_heatmap_ablation(index,paths,index_fix):#传入列表,原SPET图放在第一个
    n=len(paths)
    plt.figure(figsize=(5*(n+1),5))
    s_img=np.array(cv2.imread(paths[0]+str(index)+'.jpg',0),dtype=float)
    for i in range(n):
        if(i==0):
            continue
        img=np.array(cv2.imread(paths[i]+str(index-index_fix[i])+'.jpg',0),dtype=float)
        plt.subplot(1, n, i+1)
        plt.title(ablation[i],fontsize=30)
        lambda_res=1.
        if (i==1):
            lambda_res=1.5*lambda_res
        elif (i==2):
            lambda_res=0.8*lambda_res
        res=lambda_res*abs(s_img-img)
        #
        cbar_flag=False
        if(i==n-1):
            cbar_flag=True
        sns.heatmap(data=res,cmap=plt.get_cmap('Blues'),vmax=55,xticklabels=False,yticklabels=False)
if __name__ == '__main__':
    for i in range(60,80):
        print_heatmap_ablation(i,['./imgs/hp033-pet-720-reorient-align-resample.img/',
                           './imgs/hp033-pet-180-1-reorient-align-resample.img/',
                           './imgs/GANfake_8.img/',
                           './imgs/ablation_cut_0216.img/',
                           './imgs/cut_1000.img/'],[0,0,32,0,0])
  