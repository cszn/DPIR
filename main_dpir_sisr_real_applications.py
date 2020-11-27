import os.path
import glob
import cv2
import logging
import time

import numpy as np
from datetime import datetime
from collections import OrderedDict
import hdf5storage

import torch

from utils import utils_deblur
from utils import utils_logger
from utils import utils_model
from utils import utils_pnp as pnp
from utils import utils_sisr as sr
from utils import utils_image as util


"""
Spyder (Python 3.7)
PyTorch 1.6.0
Windows 10 or Linux
Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/DPIR
        https://github.com/cszn/IRCNN
        https://github.com/cszn/KAIR
@article{zhang2020plug,
  title={Plug-and-Play Image Restoration with Deep Denoiser Prior},
  author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint},
  year={2020}
}
% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com; homepage: https://cszn.github.io/)
by Kai Zhang (01/August/2020)

# --------------------------------------------
|--model_zoo               # model_zoo
   |--drunet_gray          # model_name, for color images
   |--drunet_color
|--testset                 # testsets
|--results                 # results
# --------------------------------------------
"""

def main():

    """
    # ----------------------------------------------------------------------------------
    # In real applications, you should set proper 
    # - "noise_level_img": from [3, 25], set 3 for clean image, try 15 for very noisy LR images
    # - "k" (or "kernel_width"): blur kernel is very important!!!  kernel_width from [0.6, 3.0]
    # to get the best performance.
    # ----------------------------------------------------------------------------------
    """
    ##############################################################################

    testset_name = 'Set3C'               # set test set,  'set5' | 'srbsd68'
    noise_level_img = 3                  # set noise level of image, from [3, 25], set 3 for clean image
    model_name = 'drunet_color' # 'ircnn_color'         # set denoiser, | 'drunet_color' | 'ircnn_gray' | 'drunet_gray' | 'ircnn_color'
    sf = 2                               # set scale factor, 1, 2, 3, 4
    iter_num = 24                        # set number of iterations, default: 24 for SISR

    # --------------------------------
    # set blur kernel
    # --------------------------------
    kernel_width_default_x1234 = [0.6, 0.9, 1.7, 2.2] # Gaussian kernel widths for x1, x2, x3, x4
    noise_level_model = noise_level_img/255.  # noise level of model
    kernel_width = kernel_width_default_x1234[sf-1]

    """
    # set your own kernel width !!!!!!!!!!
    """
    # kernel_width = 1.0


    k = utils_deblur.fspecial('gaussian', 25, kernel_width)
    k = sr.shift_pixel(k, sf)  # shift the kernel
    k /= np.sum(k)

    ##############################################################################


    show_img = False
    util.surf(k) if show_img else None
    x8 = True                            # default: False, x8 to boost performance
    modelSigma1 = 49                     # set sigma_1, default: 49
    modelSigma2 = max(sf, noise_level_model*255.)
    classical_degradation = True         # set classical degradation or bicubic degradation

    task_current = 'sr'                  # 'sr' for super-resolution
    n_channels = 1 if 'gray' in model_name else 3  # fixed
    model_zoo = 'model_zoo'              # fixed
    testsets = 'testsets'                # fixed
    results = 'results'                  # fixed
    result_name = testset_name + '_realapplications_' + task_current + '_' + model_name
    model_path = os.path.join(model_zoo, model_name+'.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------
    L_path = os.path.join(testsets, testset_name) # L_path, for Low-quality images
    E_path = os.path.join(results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # ----------------------------------------
    # load model
    # ----------------------------------------
    if 'drunet' in model_name:
        from models.network_unet import UNetRes as net
        model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for _, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)
    elif 'ircnn' in model_name:
        from models.network_dncnn import IRCNN as net
        model = net(in_nc=n_channels, out_nc=n_channels, nc=64)
        model25 = torch.load(model_path)
        former_idx = 0

    logger.info('model_name:{}, image sigma:{:.3f}, model sigma:{:.3f}'.format(model_name, noise_level_img, noise_level_model))
    logger.info('Model path: {:s}'.format(model_path))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    for idx, img in enumerate(L_paths):

        # --------------------------------
        # (1) get img_L
        # --------------------------------
        logger.info('Model path: {:s} Image: {:s}'.format(model_path, img))
        img_name, ext = os.path.splitext(os.path.basename(img))
        img_L = util.imread_uint(img, n_channels=n_channels)
        img_L = util.uint2single(img_L)
        img_L = util.modcrop(img_L, 8)  # modcrop

        # --------------------------------
        # (2) get rhos and sigmas
        # --------------------------------
        rhos, sigmas = pnp.get_rho_sigma(sigma=max(0.255/255., noise_level_model), iter_num=iter_num, modelSigma1=modelSigma1, modelSigma2=modelSigma2, w=1)
        rhos, sigmas = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device)

        # --------------------------------
        # (3) initialize x, and pre-calculation
        # --------------------------------
        x = cv2.resize(img_L, (img_L.shape[1]*sf, img_L.shape[0]*sf), interpolation=cv2.INTER_CUBIC)

        if np.ndim(x)==2:
            x = x[..., None]

        if classical_degradation:
            x = sr.shift_pixel(x, sf)
        x = util.single2tensor4(x).to(device)

        img_L_tensor, k_tensor = util.single2tensor4(img_L), util.single2tensor4(np.expand_dims(k, 2))
        [k_tensor, img_L_tensor] = util.todevice([k_tensor, img_L_tensor], device)
        FB, FBC, F2B, FBFy = sr.pre_calculate(img_L_tensor, k_tensor, sf)

        # --------------------------------
        # (4) main iterations
        # --------------------------------
        for i in range(iter_num):

            print('Iter: {} / {}'.format(i, iter_num))

            # --------------------------------
            # step 1, FFT
            # --------------------------------
            tau = rhos[i].float().repeat(1, 1, 1, 1)
            x = sr.data_solution(x, FB, FBC, F2B, FBFy, tau, sf)

            if 'ircnn' in model_name:
                current_idx = np.int(np.ceil(sigmas[i].cpu().numpy()*255./2.)-1)
    
                if current_idx != former_idx:
                    model.load_state_dict(model25[str(current_idx)], strict=True)
                    model.eval()
                    for _, v in model.named_parameters():
                        v.requires_grad = False
                    model = model.to(device)
                former_idx = current_idx

            # --------------------------------
            # step 2, denoiser
            # --------------------------------
            if x8:
                x = util.augment_img_tensor4(x, i % 8)
                
            if 'drunet' in model_name:
                x = torch.cat((x, sigmas[i].repeat(1, 1, x.shape[2], x.shape[3])), dim=1)
                x = utils_model.test_mode(model, x, mode=2, refield=64, min_size=256, modulo=16)
            elif 'ircnn' in model_name:
                x = model(x)

            if x8:
                if i % 8 == 3 or i % 8 == 5:
                    x = util.augment_img_tensor4(x, 8 - i % 8)
                else:
                    x = util.augment_img_tensor4(x, i % 8)

        # --------------------------------
        # (3) img_E
        # --------------------------------
        img_E = util.tensor2uint(x)
        util.imsave(img_E, os.path.join(E_path, img_name+'_x'+str(sf)+'_'+model_name+'.png'))

if __name__ == '__main__':

    main()
