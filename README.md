# DPIR

Abstract
----------
Recent works on plug-and-play image restoration have shown that a denoiser can implicitly serve as the image prior for
model-based methods to solve many inverse problems. Such a property induces considerable advantages for plug-and-play image
restoration (e.g., integrating the flexibility of model-based method and effectiveness of learning-based methods) when the denoiser is
discriminatively learned via deep convolutional neural network (CNN) with large modeling capacity. However, while deeper and larger
CNN models are rapidly gaining popularity, existing plug-and-play image restoration hinders its performance due to the lack of suitable
denoiser prior. In order to push the limits of plug-and-play image restoration, we set up a benchmark deep denoiser prior by training a
highly flexible and effective CNN denoiser. We then plug the deep denoiser prior as a modular part into a half quadratic splitting based
iterative algorithm to solve various image restoration problems. We, meanwhile, provide a thorough analysis of parameter setting,
intermediate results and empirical convergence to better understand the working mechanism. Experimental results on three
representative image restoration tasks, including deblurring, super-resolution and demosaicing, demonstrate that the proposed
plug-and-play image restoration with deep denoiser prior not only significantly outperforms other state-of-the-art model-based methods
but also achieves competitive or even superior performance against state-of-the-art learning-based methods.


The Deep Denoiser
----------
* Network architecture

  <img src="figs/denoiser_arch.png" width="800px"/> 

* Grayscale image denoising

  <img src="figs/grayscale_psnr.png" width="800px"/> 

* Color image denoising

  <img src="figs/color_psnr.png" width="375px"/> 


Image Deblurring
----------
* Visual comparison
  <img src="figs/deblur_1.png" width="800px"/> 


Single Image Super-Resolution
----------
* Visual comparison
  <img src="figs/sisr_1.png" width="800px"/> 


Color Image Demosaicing
----------
* PSNR
  <img src="figs/demosaic_1.png" width="800px"/> 
* Visual comparison
  <img src="figs/demosaic_2.png" width="800px"/> 


Citation
----------
```BibTex
@article{zhang2020plug,
  title={Plug-and-Play Image Restoration with Deep Denoiser Prior},
  author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint},
  year={2020}
}
```
