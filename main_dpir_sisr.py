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
   |--drunet_color         # model_name, for color images
   |--drunet_gray
|--testset                 # testsets
|--results                 # results
# --------------------------------------------


How to run:
step 1: download [drunet_gray.pth, drunet_color.pth, ircnn_gray.pth, ircnn_color.pth] from https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D
step 2: set your own testset 'testset_name' and parameter setting such as 'noise_level_img', 'iter_num'. 
step 3: 'python main_dpir_sisr.py'

"""

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    noise_level_img = 0/255.0            # set AWGN noise level for LR image, default: 0, 
    noise_level_model = noise_level_img  # setnoise level of model, default 0
    model_name = 'drunet_color'          # set denoiser, | 'drunet_color' | 'ircnn_gray' | 'drunet_gray' | 'ircnn_color'
    testset_name = 'srbsd68'             # set test set,  'set5' | 'srbsd68'
    x8 = True                            # default: False, x8 to boost performance
    test_sf = [2]                        # set scale factor, default: [2, 3, 4], [2], [3], [4]
    iter_num = 24                        # set number of iterations, default: 24 for SISR
    modelSigma1 = 49                     # set sigma_1, default: 49
    classical_degradation = True         # set classical degradation or bicubic degradation

    show_img = False                     # default: False
    save_L = True                        # save LR image
    save_E = True                        # save estimated image
    save_LEH = False                     # save zoomed LR, E and H images

    task_current = 'sr'                  # 'sr' for super-resolution
    n_channels = 1 if 'gray' in model_name else 3  # fixed
    model_zoo = 'model_zoo'              # fixed
    testsets = 'testsets'                # fixed
    results = 'results'                  # fixed
    result_name = testset_name + '_' + task_current + '_' + model_name
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

    # --------------------------------
    # load kernel
    # --------------------------------

    # kernels = hdf5storage.loadmat(os.path.join('kernels', 'Levin09.mat'))['kernels']
    if classical_degradation:
        kernels = hdf5storage.loadmat(os.path.join('kernels', 'kernels_12.mat'))['kernels']
    else:
        kernels = hdf5storage.loadmat(os.path.join('kernels', 'kernel_bicubicx234.mat'))['kernels']

    test_results_ave = OrderedDict()
    test_results_ave['psnr_sf_k'] = []
    test_results_ave['psnr_y_sf_k'] = []

    for sf in test_sf:
        border = sf
        modelSigma2 = max(sf, noise_level_model*255.)
        k_num = 8 if classical_degradation else 1

        for k_index in range(k_num):
            logger.info('--------- sf:{:>1d} --k:{:>2d} ---------'.format(sf, k_index))
            test_results = OrderedDict()
            test_results['psnr'] = []
            test_results['psnr_y'] = []

            if not classical_degradation:  # for bicubic degradation
                k_index = sf-2
            k = kernels[0, k_index].astype(np.float64)

            util.surf(k) if show_img else None

            for idx, img in enumerate(L_paths):

                # --------------------------------
                # (1) get img_L
                # --------------------------------

                img_name, ext = os.path.splitext(os.path.basename(img))
                img_H = util.imread_uint(img, n_channels=n_channels)
                img_H = util.modcrop(img_H, sf)  # modcrop

                if classical_degradation:
                    img_L = sr.classical_degradation(img_H, k, sf)
                    util.imshow(img_L) if show_img else None
                    img_L = util.uint2single(img_L)
                else:
                    img_L = util.imresize_np(util.uint2single(img_H), 1/sf)

                np.random.seed(seed=0)  # for reproducibility
                img_L += np.random.normal(0, noise_level_img, img_L.shape) # add AWGN

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

                    # --------------------------------
                    # step 1, FFT
                    # --------------------------------

                    tau = rhos[i].float().repeat(1, 1, 1, 1)
                    x = sr.data_solution(x.float(), FB, FBC, F2B, FBFy, tau, sf)

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
                        x = torch.cat((x, sigmas[i].float().repeat(1, 1, x.shape[2], x.shape[3])), dim=1)
                        x = utils_model.test_mode(model, x, mode=2, refield=32, min_size=256, modulo=16)
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

                if save_E:
                    util.imsave(img_E, os.path.join(E_path, img_name+'_x'+str(sf)+'_k'+str(k_index)+'_'+model_name+'.png'))

                if n_channels == 1:
                    img_H = img_H.squeeze()

                # --------------------------------
                # (4) img_LEH
                # --------------------------------

                img_L = util.single2uint(img_L).squeeze()

                if save_LEH:
                    k_v = k/np.max(k)*1.0
                    if n_channels==1:
                        k_v = util.single2uint(k_v)
                    else:
                        k_v = util.single2uint(np.tile(k_v[..., np.newaxis], [1, 1, n_channels]))
                    k_v = cv2.resize(k_v, (3*k_v.shape[1], 3*k_v.shape[0]), interpolation=cv2.INTER_NEAREST)
                    img_I = cv2.resize(img_L, (sf*img_L.shape[1], sf*img_L.shape[0]), interpolation=cv2.INTER_NEAREST)
                    img_I[:k_v.shape[0], -k_v.shape[1]:, ...] = k_v
                    img_I[:img_L.shape[0], :img_L.shape[1], ...] = img_L
                    util.imshow(np.concatenate([img_I, img_E, img_H], axis=1), title='LR / Recovered / Ground-truth') if show_img else None
                    util.imsave(np.concatenate([img_I, img_E, img_H], axis=1), os.path.join(E_path, img_name+'_x'+str(sf)+'_k'+str(k_index)+'_LEH.png'))

                if save_L:
                    util.imsave(img_L, os.path.join(E_path, img_name+'_x'+str(sf)+'_k'+str(k_index)+'_LR.png'))

                psnr = util.calculate_psnr(img_E, img_H, border=border)
                test_results['psnr'].append(psnr)
                logger.info('{:->4d}--> {:>10s} -- sf:{:>1d} --k:{:>2d} PSNR: {:.2f}dB'.format(idx+1, img_name+ext, sf, k_index, psnr))

                if n_channels == 3:
                    img_E_y = util.rgb2ycbcr(img_E, only_y=True)
                    img_H_y = util.rgb2ycbcr(img_H, only_y=True)
                    psnr_y = util.calculate_psnr(img_E_y, img_H_y, border=border)
                    test_results['psnr_y'].append(psnr_y)

            # --------------------------------
            # Average PSNR for all kernels
            # --------------------------------

            ave_psnr_k = sum(test_results['psnr']) / len(test_results['psnr'])
            logger.info('------> Average PSNR(RGB) of ({}) scale factor: ({}), kernel: ({}) sigma: ({:.2f}): {:.2f} dB'.format(testset_name, sf, k_index, noise_level_model, ave_psnr_k))
            test_results_ave['psnr_sf_k'].append(ave_psnr_k)

            if n_channels == 3:  # RGB image
                ave_psnr_y_k = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
                logger.info('------> Average PSNR(Y) of ({}) scale factor: ({}), kernel: ({}) sigma: ({:.2f}): {:.2f} dB'.format(testset_name, sf, k_index, noise_level_model, ave_psnr_y_k))
                test_results_ave['psnr_y_sf_k'].append(ave_psnr_y_k)

    # ---------------------------------------
    # Average PSNR for all sf and kernels
    # ---------------------------------------

    ave_psnr_sf_k = sum(test_results_ave['psnr_sf_k']) / len(test_results_ave['psnr_sf_k'])
    logger.info('------> Average PSNR of ({}) {:.2f} dB'.format(testset_name, ave_psnr_sf_k))
    if n_channels == 3:
        ave_psnr_y_sf_k = sum(test_results_ave['psnr_y_sf_k']) / len(test_results_ave['psnr_y_sf_k'])
        logger.info('------> Average PSNR of ({}) {:.2f} dB'.format(testset_name, ave_psnr_y_sf_k))

if __name__ == '__main__':

    main()
