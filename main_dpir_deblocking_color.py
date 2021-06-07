import os.path
import logging
import numpy as np
from collections import OrderedDict

import torch
from utils import utils_logger
from utils import utils_image as util
import cv2




def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    model_name = 'drunet_color'
    quality_factors = [10, 20, 30, 40]
    testset_name = 'LIVE1'               # test set,  'LIVE1'
    need_degradation = True              # default: True

    task_current = 'db'       # 'dn' for deblocking
    n_channels = 3            # fixed
    model_pool = 'model_zoo'  # fixed
    testsets = 'testsets'     # fixed
    results = 'results'       # fixed
    noise_level_img = 0       # fixed: 0, noise level for LR image
    result_name = testset_name + '_' + model_name + '_' + task_current
    border = 0     # shave boader to calculate PSNR and SSIM

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------
    L_path = os.path.join(testsets, testset_name) # L_path, for Low-quality images
    E_path = os.path.join(results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------
    model_path = os.path.join('model_zoo', 'drunet_deblocking_color.pth')
    from models.network_unet import UNetRes as net
    model = net(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose', bias=False)  # define network

    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
        
    model = model.to(device)
    logger.info('Model path: {:s}'.format(model_path))
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))
    L_paths = util.get_image_paths(L_path)

    for quality_factor in quality_factors:

        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['ssim'] = []

        logger.info('model_name:{}, quality factor:{}'.format(model_name, quality_factor))
        logger.info(L_path)

        for idx, img in enumerate(L_paths):
    
            # ------------------------------------
            # (1) img_L
            # ------------------------------------
            img_name, ext = os.path.splitext(os.path.basename(img))
            logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
            img_L = util.imread_uint(img, n_channels=n_channels)
            img_H = img_L.copy()

            # ------------------------------------
            # Do the JPEG compression
            # ------------------------------------
            if need_degradation:
                img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2BGR)
                result, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
                img_L = cv2.imdecode(encimg, 1)
                img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)

            img_L = util.uint2tensor4(img_L)

            noise_level = (100-quality_factor)/100.0
            noise_level = torch.FloatTensor([noise_level])
            noise_level_map = torch.ones((1,1, img_L.shape[2], img_L.shape[3])).mul_(noise_level).float()
            img_L = torch.cat((img_L, noise_level_map), 1)

            img_L = img_L.to(device)

            # ------------------------------------
            # (2) img_E
            # ------------------------------------
            img_E = model(img_L)
            img_E = util.tensor2uint(img_E)

            if need_degradation:

                img_H = img_H.squeeze()
                # --------------------------------
                # PSNR and SSIM
                # --------------------------------
                psnr = util.calculate_psnr(img_E, img_H, border=border)
                ssim = util.calculate_ssim(img_E, img_H, border=border)
                test_results['psnr'].append(psnr)
                test_results['ssim'].append(ssim)
                logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name+ext, psnr, ssim))

            # ------------------------------------
            # save results
            # ------------------------------------
            util.imsave(img_E, os.path.join(E_path, img_name+'_'+model_name+'_'+str(quality_factor)+'.png'))

        if need_degradation:
            ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
            ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
            logger.info('Average PSNR/SSIM(RGB) - {} - qf{} --PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, quality_factor, ave_psnr, ave_ssim))


if __name__ == '__main__':

    main()
