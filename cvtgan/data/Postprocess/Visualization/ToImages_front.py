
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 17:36:22 2022

@author: AruZeng
"""

import numpy as np
import SimpleITK as sitk
from medpy.io import load
import scipy.io as sio
from glob import glob
import cv2
import os
import copy as cp
from PIL import Image
def MatrixToImage(data):
    if(data.max()>2):
        data=(data-data.min())/(data.max()-data.min())
    print(data.max(),data.min())
    data=data*255
    data=np.flipud(data)
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im
def ToImages(root,fix):
    all_names=[]
    for root, dirs, files in os.walk(root):
        all_names=(files)
    #
    all_name=[]
    for i in all_names:
        if os.path.splitext(i)[1] == ".img":
            all_name.append(i)
    print(all_name)
    for file in all_name:
        image_path = os.path.join(root,file)
        image,h=load(image_path)
        image=np.array(image)
        image = np.moveaxis(image,[0,1,2],[2,1,0])
        mask = np.zeros((128,128,128))
        #
        folder = os.path.exists('./imgs_front/' +fix+ file)
        if not folder:
            os.makedirs('./imgs_front/' +fix +file) 
        for j in range(128):
            img = MatrixToImage(image[:, j, :])
            img.save('./imgs_front/' +fix+ file+'/'+ str(j) + '.jpg')
#
def ToImages1(root):
    all_names=[]
    for root, dirs, files in os.walk(root):
        all_names=(files)
    #
    all_name=[]
    for i in all_names:
        if os.path.splitext(i)[1] == ".img":
            all_name.append(i)
    print(all_name)
    for file in all_name:
        image_path = os.path.join(root,file)
        image,h=load(image_path)
        image=np.array(image)
        image = np.moveaxis(image,[0,1,2],[2,1,0])
        #image=image*2
        mask = np.zeros((128,128,128))
        #
        folder = os.path.exists('./imgs_front/' + file)
        if not folder:
            os.makedirs('./imgs_front/' + file) 
        for j in range(128):
            img = MatrixToImage(image[:, j, :])
            img.save('./imgs_front/' + file+'/'+ str(j) + '.jpg')
if __name__ == '__main__':
    ToImages('./data/predict_results','')
    ToImages('./results/transformergan/fakehp','transgan')
    ToImages('./results/unet/fakehp','unet')
    ToImages('./data/sclip','')
    ToImages('./data/lclip','')