import numpy as np
from skimage import io
import SimpleITK as sitk
import scipy.io as sio
from medpy.io import load
from glob import glob
import os
import copy as cp
import skimage.transform as k

if __name__ == '__main__':

    mainPath = '/home/user/lym/ARN/PET/clinicaldata/datasets/hp*'
    savePath = '/home/user/lym/ARN/PET/dose-2D-007/'
    labelPath = glob(mainPath)

    for eachPath in labelPath:
        fn = os.path.basename(eachPath)
        print(fn)

        f = open(eachPath+'/'+fn+'-t1-reorient-alignToPet-strip-align-resample.img', 'r')
        img = sitk.ReadImage(eachPath+'/'+fn+'-t1-reorient-alignToPet-strip-align-resample.img')
        t1, h = load(eachPath+'/'+fn+'-t1-reorient-alignToPet-strip-align-resample.img')
        t1 = np.array(t1)
        t1 = np.moveaxis(t1,[0,1,2],[-2,-1,-3])

        newLabel = np.zeros((128,128,128))

        for j in xrange(128):
            eachMask = cp.copy(t1[j,:,:])
            posi = np.argwhere(eachMask!=0)

            labelAll = np.zeros((128,128))

            if len(posi)==0:
                continue

            image = sio.loadmat(savePath+fn+'-'+str(j)+'_fake_B.mat')
            labelAll = image['img']
            labelAll = (labelAll+1)/2

            labelAll=labelAll*(eachMask!=0)
            labelAll = np.moveaxis(labelAll,[0,1,2],[0,2,1])

            newLabel[j,:,:]=cp.copy(labelAll)

        savImg = sitk.GetImageFromArray(newLabel)
        sitk.WriteImage(savImg, eachPath+'/'+fn+'-pet-720-2D-fake.img')
