# -*- coding: utf-8 -*-
import numpy as np
from utils import utils_image as util 

'''
modified by Kai Zhang (github: https://github.com/cszn)
03/03/2019
'''


# --------------------------------
# get rho and sigma
# --------------------------------
def get_rho_sigma(sigma=2.55/255, iter_num=15, modelSigma2=2.55):
    '''
    Kai Zhang (github: https://github.com/cszn)
    03/03/2019
    '''
    modelSigma1 = 49.0
    modelSigmaS = np.logspace(np.log10(modelSigma1), np.log10(modelSigma2), iter_num)
    sigmas = modelSigmaS/255.
    mus = list(map(lambda x: (sigma**2)/(x**2)/3, sigmas))
    rhos = mus
    return rhos, sigmas


def shepard_initialize(image, measurement_mask, window=5, p=2):
    wing = np.floor(window/2).astype(int) # Length of each "wing" of the window.
    h, w = image.shape[0:2]
    ch = 3 if image.ndim == 3 and image.shape[-1] == 3 else 1
    x = np.copy(image) # ML initialization
    for i in range(h):
        i_lower_limit = -np.min([wing, i])
        i_upper_limit = np.min([wing, h-i-1])
        for j in range(w):
           if measurement_mask[i, j] == 0: # checking if there's a need to interpolate
               j_lower_limit = -np.min([wing, j])
               j_upper_limit = np.min([wing, w-j-1])

               count = 0 # keeps track of how many measured pixels are withing the neighborhood
               sum_IPD = 0
               interpolated_value = 0

               num_zeros = window**2
               IPD = np.zeros([num_zeros])
               pixel = np.zeros([num_zeros,ch])

               for neighborhood_i in range(i+i_lower_limit, i+i_upper_limit):
                   for neighborhood_j in range(j+j_lower_limit, j+j_upper_limit):
                      if measurement_mask[neighborhood_i, neighborhood_j] == 1:
                          # IPD: "inverse pth-power distance".
                          IPD[count] = 1.0/((neighborhood_i - i)**p + (neighborhood_j - j)**p)
                          sum_IPD = sum_IPD + IPD[count]
                          pixel[count] = image[neighborhood_i, neighborhood_j]
                          count = count + 1

               for c in range(count):
                   weight = IPD[c]/sum_IPD
                   interpolated_value = interpolated_value + weight*pixel[c]
               x[i, j] = interpolated_value

    return x


if __name__ == '__main__':
    # image path & sampling ratio
    import matplotlib.pyplot as mplot
    import matplotlib.image as mpimg
    Im = mpimg.imread('test.bmp')
    #Im = Im[:,:,1]
    Im = np.squeeze(Im)

    SmpRatio = 0.2
    # creat mask
    mask_Array = np.random.rand(Im.shape[0],Im.shape[1])
    mask_Array = (mask_Array < SmpRatio)
    print(mask_Array.dtype)

    # sampled image
    print('The sampling ratio is', SmpRatio)
    Im_sampled = np.multiply(np.expand_dims(mask_Array,2), Im)
    util.imshow(Im_sampled)
    
    a = shepard_initialize(Im_sampled.astype(np.float32), mask_Array, window=9)
    a = np.clip(a,0,255)


    print(a.dtype)
    
    
    util.imshow(np.concatenate((a,Im_sampled),1)/255.0)
    util.imsave(np.concatenate((a,Im_sampled),1),'a.png')

    


