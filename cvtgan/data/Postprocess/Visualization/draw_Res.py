# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 14:03:41 2022

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
    #print(data.max(),data.min())
    data=data*255
    #data=np.flipud(data)
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im
img1=np.array(cv2.imread('high.jpg',0),dtype=float)
img2=np.array(cv2.imread('low.jpg',0),dtype=float)
res=abs(img1-img2)
res=res*2
print(res.shape)
res = Image.fromarray(res.astype(np.uint8))
res.save('residual.jpg')