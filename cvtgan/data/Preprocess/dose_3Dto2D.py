import numpy as np
import SimpleITK as sitk
from medpy.io import load
import scipy.io as sio
from glob import glob
import os
import copy as cp
from PIL import Image


def MatrixToImage(data):
	new_im = Image.fromarray(data.astype(np.uint8))
	return new_im


if __name__ == '__main__':

	mainPath = '/home/user/lym/ARN/PET/clinicaldata/datasets/hp*'  ## hp-->PE
	savePath = '/home/user/lym/ARN/PET/dose-2D-007/'

	
	labelPath = glob(mainPath)

	for eachPath in labelPath:
		fn = os.path.basename(eachPath)
		print(fn)

		

		im1n,h = load( eachPath+'/'+fn+'-pet-180-1-reorient-align-resample.img')
		im1n = np.array(im1n)
		im1n = np.moveaxis(im1n,[0,1,2],[-2,-1,-3])
		
		im2n,h = load( eachPath+'/'+fn+'-pet-720-reorient-align-resample.img')
		im2n = np.array(im2n)
		# im2n = np.moveaxis(im2n,[0,1,2],[-2,-1,-3])
		
		t1,h = load( eachPath+'/'+fn+'-t1-reorient-alignToPet-strip-align-resample.img')
		t1 = np.array(t1)
		t1 = np.moveaxis(t1,[0,1,2],[-2,-1,-3])
		
		mask = np.zeros((128,128,128))
		#
		posi1=np.argwhere(t1!=0)
		# for po in posi1:
		# 	mask[po[0],po[1],po[2]]=1
		#
		#
		# data = cp.copy(im1n)*mask
		# amin = np.amin(data[np.nonzero(data)])
		# data = data-amin
		# amax = np.amax(data)
		# im1n = cp.copy((im1n-amin)/amax* mask-0.5)*2
		#
		# data = cp.copy(im2n)*mask
		# amin = np.amin(data[np.nonzero(data)])
		# data = data-amin
		# amax = np.amax(data)
		# im2n = cp.copy((im2n-amin)/amax* mask-0.5)*2
		


		for j in range(128):
			img = MatrixToImage(im1n[j, :, :])
			img.save(savePath + 'imgs/' + fn + '-' + str(j) + '.jpg')
						
			# eachMask = cp.copy(mask[j,:,:])
			# posi = np.argwhere(eachMask==1)
			#
			# newImgA = np.zeros((1,128,128))
			# newImgB = np.zeros((1,128,128))
			#
			#
			#
			# if len(posi)==0:
			# 	continue
			#
			#
			# newImgA[0,:,:] = cp.copy(im1n[j,:,:])
			#
			# newImgB[0,:,:]  = cp.copy(im2n[j,:,:])


			


			# newImgAB = np.concatenate([newImgA, newImgB],axis=1)

			
			# if fn=='hp007':
			# 	sio.savemat(savePath+'test/'+fn+'-'+str(j)+'.mat', {'img':newImgAB})
			# else:
			# 	sio.savemat(savePath+'train/'+fn+'-'+str(j)+'.mat', {'img':newImgAB})
			
