# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 10:38:30 2022

@author: AruZeng
"""
from torch.autograd import Variable
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
def create3DsobelFilter():
    num_1, num_2, num_3 = np.zeros((3,3))
    num_1 = [[1., 2., 1.],
             [2., 4., 2.],
             [1., 2., 1.]]
    num_2 = [[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]]
    num_3 = [[-1., -2., -1.],
             [-2., -4., -2.],
             [-1., -2., -1.]]
    sobelFilter = np.zeros((3,1,3,3,3))

    sobelFilter[0,0,0,:,:] = num_1
    sobelFilter[0,0,1,:,:] = num_2
    sobelFilter[0,0,2,:,:] = num_3
    sobelFilter[1,0,:,0,:] = num_1
    sobelFilter[1,0,:,1,:] = num_2
    sobelFilter[1,0,:,2,:] = num_3
    sobelFilter[2,0,:,:,0] = num_1
    sobelFilter[2,0,:,:,1] = num_2
    sobelFilter[2,0,:,:,2] = num_3
    return Variable(torch.from_numpy(sobelFilter).type(torch.cuda.FloatTensor))

def sobelLayer(input):
    pad = nn.ConstantPad3d((1,1,1,1,1,1),-1)
    kernel = create3DsobelFilter()
    act = nn.Tanh()
    paded = pad(input)
    fake_sobel = F.conv3d(paded, kernel, padding = 0, groups = 1)/4
    n,c,h,w,l = fake_sobel.size()
    fake = torch.norm(fake_sobel,2,1,True)/c*3
    fake_out = act(fake)*2-1

    return fake_out
