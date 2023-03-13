# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 19:37:52 2022

@author: AruZeng
"""
import SimpleITK as sitk
import torch
import numpy as np
from medpy.io import load
def detect_edge_1(image):
    image_float = sitk.Cast(image, sitk.sitkFloat32)
    sobel_op = sitk.SobelEdgeDetectionImageFilter()
    sobel_sitk = sobel_op.Execute(image_float)
    sobel_sitk = sitk.Cast(sobel_sitk, sitk.sitkInt16)
    return sobel_sitk
image_l,h=load('./data/clinical/val_l.img')
image=image_l
image=(image-image.min())/(image.max()-image.min())
savImg = sitk.GetImageFromArray(image)
sitk.WriteImage(detect_edge_1(savImg),'./data/edge.mha')