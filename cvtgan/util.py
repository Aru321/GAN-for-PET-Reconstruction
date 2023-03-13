from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
import SimpleITK as sitk
import scipy.io as sio

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()[0,0,30:33,:,:]
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = (image_numpy + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def tensor2array(image_tensor):
    image_numpy = image_tensor.cpu().numpy()
    # print(image_numpy.shape)
    # img_numpy = np.array(image_numpy).reshape((1,1,64,64,64))[0,0,:,:,:]
    return image_numpy[0,0,:,:,:]

def tensor2array_labels(image_tensor):
    image_numpy = image_tensor.cpu().numpy()
    # print(image_numpy.shape)
    # img_numpy = np.array(image_numpy).reshape((1,1,64,64,64))[0,0,:,:,:]
    return image_numpy[0,:,:,:,:]


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):

    savImg = sitk.GetImageFromArray(image_numpy[:,:,:])
    sitk.WriteImage(savImg, image_path)
    # savImg = sitk.GetImageFromArray(image_numpy[1,:,:,:])
    # sitk.WriteImage(image_path, eachPath +'/' + fn +'_2n_fake.nii.gz')
    # savImg = sitk.GetImageFromArray(image_numpy[2,:,:,:])
    # sitk.WriteImage(image_path, eachPath +'/' + fn +'_4n_fake.nii.gz')
def save_labels(image_numpy, image_path):


    sio.savemat(image_path, {'label':image_numpy})

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm3d') != -1 or  classname.find('InstanceNorm3d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)