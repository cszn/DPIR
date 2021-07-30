import os.path
import cv2
import logging

import numpy as np
from collections import OrderedDict

import torch

from utils import utils_model
from utils import utils_mosaic
from utils import utils_logger
from utils import utils_pnp as pnp
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

How to run:
step 1: download [drunet_color.pth, ircnn_color.pth] from https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D
step 2: set your own testset 'testset_name' and parameter setting such as 'noise_level_model', 'iter_num'. 
step 3: 'python main_dpir_demosaick.py'

"""

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    noise_level_img = 0/255.0            # set AWGN noise level for LR image, default: 0
    noise_level_model = noise_level_img  # set noise level of model, default: 0
    model_name = 'ircnn_color'           # set denoiser, 'drunet_color' | 'ircnn_color'
    testset_name = 'Set18'               # set testing set,  'set18' | 'set24'
    x8 = True                            # set PGSE to boost performance, default: True
    iter_num = 40                        # set number of iterations, default: 40 for demosaicing
    modelSigma1 = 49                     # set sigma_1, default: 49
    modelSigma2 = max(0.6, noise_level_model*255.) # set sigma_2, default
    matlab_init = True

    show_img = False                     # default: False
    save_L = True                        # save LR image
    save_E = True                        # save estimated image
    save_LEH = False                     # save zoomed LR, E and H images
    border = 10                          # default 10 for demosaicing

    task_current = 'dm'                  # 'dm' for demosaicing
    n_channels = 3                       # fixed
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

    test_results = OrderedDict()
    test_results['psnr'] = []

    for idx, img in enumerate(L_paths):

        # --------------------------------
        # (1) get img_H and img_L
        # --------------------------------

        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img))
        img_H = util.imread_uint(img, n_channels=n_channels)
        CFA, CFA4, mosaic, mask = utils_mosaic.mosaic_CFA_Bayer(img_H)

        # --------------------------------
        # (2) initialize x
        # --------------------------------

        if matlab_init:  # matlab demosaicing for initialization
            CFA4 = util.uint2tensor4(CFA4).to(device)
            x = utils_mosaic.dm_matlab(CFA4)
        else:
            x = cv2.cvtColor(CFA, cv2.COLOR_BAYER_BG2RGB_EA)
            x = util.uint2tensor4(x).to(device)

        img_L = util.tensor2uint(x)
        y = util.uint2tensor4(mosaic).to(device)

        util.imshow(img_L) if show_img else None
        mask = util.single2tensor4(mask.astype(np.float32)).to(device)

        # --------------------------------
        # (3) get rhos and sigmas
        # --------------------------------

        rhos, sigmas = pnp.get_rho_sigma(sigma=max(0.255/255., noise_level_img), iter_num=iter_num, modelSigma1=modelSigma1, modelSigma2=modelSigma2, w=1.0)
        rhos, sigmas = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device)

        # --------------------------------
        # (4) main iterations
        # --------------------------------

        for i in range(iter_num):

            # --------------------------------
            # step 1, closed-form solution
            # --------------------------------

            x = (y+rhos[i].float()*x).div(mask+rhos[i])

            # --------------------------------
            # step 2, denoiser
            # --------------------------------

            if 'ircnn' in model_name:
                current_idx = np.int(np.ceil(sigmas[i].cpu().numpy()*255./2.)-1)
                if current_idx != former_idx:
                    model.load_state_dict(model25[str(current_idx)], strict=True)
                    model.eval()
                    for _, v in model.named_parameters():
                        v.requires_grad = False
                    model = model.to(device)
                former_idx = current_idx

            x = torch.clamp(x, 0, 1)
            if x8:
                x = util.augment_img_tensor4(x, i % 8)

            if 'drunet' in model_name:
                x = torch.cat((x, sigmas[i].float().repeat(1, 1, x.shape[2], x.shape[3])), dim=1)
                x = utils_model.test_mode(model, x, mode=2, refield=32, min_size=256, modulo=16)
                # x = model(x)
            elif 'ircnn' in model_name:
                x = model(x)

            if x8:
                if i % 8 == 3 or i % 8 == 5:
                    x = util.augment_img_tensor4(x, 8 - i % 8)
                else:
                    x = util.augment_img_tensor4(x, i % 8)

        x[mask.to(torch.bool)] = y[mask.to(torch.bool)]

        # --------------------------------
        # (4) img_E
        # --------------------------------

        img_E = util.tensor2uint(x)
        psnr = util.calculate_psnr(img_E, img_H, border=border)
        test_results['psnr'].append(psnr)
        logger.info('{:->4d}--> {:>10s} -- PSNR: {:.2f}dB'.format(idx, img_name+ext, psnr))

        if save_E:
            util.imsave(img_E, os.path.join(E_path, img_name+'_'+model_name+'.png'))

        if save_L:
            util.imsave(img_L, os.path.join(E_path, img_name+'_L.png'))

        if save_LEH:
            util.imsave(np.concatenate([img_L, img_E, img_H], axis=1), os.path.join(E_path, img_name+model_name+'_LEH.png'))

    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    logger.info('------> Average PSNR(RGB) of ({}) is : {:.2f} dB'.format(testset_name,  ave_psnr))


if __name__ == '__main__':

    main()
