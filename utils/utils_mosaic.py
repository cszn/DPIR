# -*- coding: utf-8 -*-
import numpy as np
from utils import utils_image as util 
#import utils_image as util 
import cv2
import torch
import torch.nn as nn


'''
modified by Kai Zhang (github: https://github.com/cszn)
03/03/2019
'''
def dm(imgs):
    """ bilinear demosaicking
    Args:
        imgs: Nx4xW/2xH/2

    Returns:
        output: Nx3xWxH
    """
    k_r = 1/4 * torch.FloatTensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]]).type_as(imgs)
    k_g = 1/4 * torch.FloatTensor([[0, 1, 0], [1, 4, 1], [0, 1, 0]]).type_as(imgs)
    k = torch.stack((k_r,k_g,k_r), dim=0).unsqueeze(1)

    rgb = torch.zeros(imgs.size(0), 3, imgs.size(2)*2, imgs.size(3)*2).type_as(imgs)
    rgb[:, 0, 0::2, 0::2] = imgs[:, 0, :, :]
    rgb[:, 1, 0::2, 1::2] = imgs[:, 1, :, :]
    rgb[:, 1, 1::2, 0::2] = imgs[:, 2, :, :]
    rgb[:, 2, 1::2, 1::2] = imgs[:, 3, :, :]

    rgb = nn.functional.pad(rgb, (1, 1, 1, 1), mode='circular')
    rgb = nn.functional.conv2d(rgb, k, groups=3, padding=0, bias=None)

    return rgb


def dm_matlab(imgs):
    """ matlab demosaicking
    Args:
        imgs: Nx4xW/2xH/2

    Returns:
        output: Nx3xWxH
    """

    kgrb = 1/8*torch.FloatTensor([[0, 0, -1, 0, 0],
                                    [0, 0, 2, 0, 0],
                                    [-1, 2, 4, 2, -1],
                                    [0, 0, 2, 0, 0],
                                    [0, 0, -1, 0, 0]]).type_as(imgs)
    krbg0 = 1/8*torch.FloatTensor([[0, 0, 1/2, 0, 0],
                                      [0, -1, 0, -1, 0],
                                      [-1, 4, 5, 4, -1],
                                      [0, -1, 0, -1, 0],
                                      [0, 0, 1/2, 0, 0]]).type_as(imgs)
    krbg1 = krbg0.t()
    krbbr = 1/8*torch.FloatTensor([[0, 0, -3/2, 0, 0],
                                     [0, 2, 0, 2, 0],
                                     [-3/2, 0, 6, 0, -3/2],
                                     [0, 2, 0, 2, 0],
                                     [0, 0, -3/2, 0, 0]]).type_as(imgs)
    
    k = torch.stack((kgrb, krbg0, krbg1, krbbr), 0).unsqueeze(1)

    cfa = torch.zeros(imgs.size(0), 1, imgs.size(2)*2, imgs.size(3)*2).type_as(imgs)
    cfa[:, 0, 0::2, 0::2] = imgs[:, 0, :, :]
    cfa[:, 0, 0::2, 1::2] = imgs[:, 1, :, :]
    cfa[:, 0, 1::2, 0::2] = imgs[:, 2, :, :]
    cfa[:, 0, 1::2, 1::2] = imgs[:, 3, :, :]
    rgb = cfa.repeat(1, 3, 1, 1)

    cfa = nn.functional.pad(cfa, (2, 2, 2, 2), mode='reflect')
    conv_cfa = nn.functional.conv2d(cfa, k, padding=0, bias=None)

    # fill G
    rgb[:, 1, 0::2, 0::2] = conv_cfa[:, 0, 0::2, 0::2]
    rgb[:, 1, 1::2, 1::2] = conv_cfa[:, 0, 1::2, 1::2]

    # fill R
    rgb[:, 0, 0::2, 1::2] = conv_cfa[:, 1, 0::2, 1::2]
    rgb[:, 0, 1::2, 0::2] = conv_cfa[:, 2, 1::2, 0::2]
    rgb[:, 0, 1::2, 1::2] = conv_cfa[:, 3, 1::2, 1::2]
    
    # fill B
    rgb[:, 2, 0::2, 1::2] = conv_cfa[:, 2, 0::2, 1::2]
    rgb[:, 2, 1::2, 0::2] = conv_cfa[:, 1, 1::2, 0::2]
    rgb[:, 2, 0::2, 0::2] = conv_cfa[:, 3, 0::2, 0::2]

    return rgb


def tstack(a):  # cv2.merge()
    a = np.asarray(a)
    return np.concatenate([x[..., np.newaxis] for x in a], axis=-1)


def tsplit(a):  # cv2.split()
    a = np.asarray(a)
    return np.array([a[..., x] for x in range(a.shape[-1])])


def masks_CFA_Bayer(shape):
    pattern = 'RGGB'
    channels = dict((channel, np.zeros(shape)) for channel in 'RGB')
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = 1
    return tuple(channels[c].astype(bool) for c in 'RGB')


def mosaic_CFA_Bayer(RGB):
    R_m, G_m, B_m = masks_CFA_Bayer(RGB.shape[0:2])
    mask = np.concatenate((R_m[..., np.newaxis], G_m[..., np.newaxis], B_m[..., np.newaxis]), axis=-1)
    # mask = tstack((R_m, G_m, B_m))
    mosaic = np.multiply(mask, RGB)  # mask*RGB
    CFA = mosaic.sum(2).astype(np.uint8)

    CFA4 = np.zeros((RGB.shape[0]//2, RGB.shape[1]//2, 4), dtype=np.uint8)
    CFA4[:, :, 0] = CFA[0::2, 0::2]
    CFA4[:, :, 1] = CFA[0::2, 1::2]
    CFA4[:, :, 2] = CFA[1::2, 0::2]
    CFA4[:, :, 3] = CFA[1::2, 1::2]

    return CFA, CFA4, mosaic, mask


if __name__ == '__main__':

    Im = util.imread_uint('test.bmp', 3)

    CFA, CFA4, mosaic, mask = mosaic_CFA_Bayer(Im)
    convertedImage = cv2.cvtColor(CFA, cv2.COLOR_BAYER_BG2RGB_EA)

    util.imshow(CFA)
    util.imshow(mosaic)
    util.imshow(mask.astype(np.float32))
    util.imshow(convertedImage)

    util.imsave(mask.astype(np.float32)*255,'bayer.png')
    


