# -*- coding: utf-8 -*-
import numpy as np
from scipy import fftpack
import torch
from scipy import ndimage
from utils import utils_image as util
from scipy.interpolate import interp2d
from scipy import signal
import scipy.stats as ss
import scipy.io as io
import scipy

'''
modified by Kai Zhang (github: https://github.com/cszn)
03/03/2019
'''


'''
# =================
# pytorch
# =================
'''


def splits(a, sf):
    '''split a into sfxsf distinct blocks

    Args:
        a: NxCxWxHx2
        sf: split factor

    Returns:
        b: NxCx(W/sf)x(H/sf)x2x(sf^2)
    '''
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=5)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=5)
    return b


def c2c(x):
    return torch.from_numpy(np.stack([np.float32(x.real), np.float32(x.imag)], axis=-1))


def r2c(x):
    # convert real to complex
    return torch.stack([x, torch.zeros_like(x)], -1)


def cdiv(x, y):
    # complex division
    a, b = x[..., 0], x[..., 1]
    c, d = y[..., 0], y[..., 1]
    cd2 = c**2 + d**2
    return torch.stack([(a*c+b*d)/cd2, (b*c-a*d)/cd2], -1)


def crdiv(x, y):
    # complex/real division
    a, b = x[..., 0], x[..., 1]
    return torch.stack([a/y, b/y], -1)


def csum(x, y):
    # complex + real
    return torch.stack([x[..., 0] + y, x[..., 1]], -1)


def cabs(x):
    # modulus of a complex number
    return torch.pow(x[..., 0]**2+x[..., 1]**2, 0.5)


def cabs2(x):
    return x[..., 0]**2+x[..., 1]**2


def cmul(t1, t2):
    '''complex multiplication

    Args:
        t1: NxCxHxWx2, complex tensor
        t2: NxCxHxWx2

    Returns:
        output: NxCxHxWx2
    '''
    real1, imag1 = t1[..., 0], t1[..., 1]
    real2, imag2 = t2[..., 0], t2[..., 1]
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1)


def cconj(t, inplace=False):
    '''complex's conjugation

    Args:
        t: NxCxHxWx2

    Returns:
        output: NxCxHxWx2
    '''
    c = t.clone() if not inplace else t
    c[..., 1] *= -1
    return c


def rfft(t):
    # Real-to-complex Discrete Fourier Transform
    return torch.rfft(t, 2, onesided=False)


def irfft(t):
    # Complex-to-real Inverse Discrete Fourier Transform
    return torch.irfft(t, 2, onesided=False)


def fft(t):
    # Complex-to-complex Discrete Fourier Transform
    return torch.fft(t, 2)


def ifft(t):
    # Complex-to-complex Inverse Discrete Fourier Transform
    return torch.ifft(t, 2)


def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.

    Args:
        psf: NxCxhxw
        shape: [H, W]

    Returns:
        otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = torch.rfft(otf, 2, onesided=False)
    n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    return otf


def upsample(x, sf=3):
    '''s-fold upsampler

    Upsampling the spatial size by filling the new entries with zeros

    x: tensor image, NxCxWxH
    '''
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*sf, x.shape[3]*sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def downsample(x, sf=3):
    '''s-fold downsampler

    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[..., st::sf, st::sf]


def data_solution(x, FB, FBC, F2B, FBFy, alpha, sf):
    FR = FBFy + torch.rfft(alpha*x, 2, onesided=False)
    x1 = cmul(FB, FR)
    FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)
    invW = torch.mean(splits(F2B, sf), dim=-1, keepdim=False)
    invWBR = cdiv(FBR, csum(invW, alpha))
    FCBinvWBR = cmul(FBC, invWBR.repeat(1, 1, sf, sf, 1))
    FX = (FR-FCBinvWBR)/alpha.unsqueeze(-1)
    Xest = torch.irfft(FX, 2, onesided=False)
    return Xest


def pre_calculate(x, k, sf):
    '''
    Args:
        x: NxCxHxW, LR input
        k: NxCxhxw
        sf: integer

    Returns:
        FB, FBC, F2B, FBFy
        will be reused during iterations
    '''
    w, h = x.shape[-2:]
    FB = p2o(k, (w*sf, h*sf))
    FBC = cconj(FB, inplace=False)
    F2B = r2c(cabs2(FB))
    STy = upsample(x, sf=sf)
    FBFy = cmul(FBC, torch.rfft(STy, 2, onesided=False))
    return FB, FBC, F2B, FBFy


'''
# =================
PyTorch
# =================
'''


def real2complex(x):
    return torch.stack([x, torch.zeros_like(x)], -1)


def modcrop(img, sf):
    '''
    img: tensor image, NxCxWxH or CxWxH or WxH
    sf: scale factor
    '''
    w, h = img.shape[-2:]
    im = img.clone()
    return im[..., :w - w % sf, :h - h % sf]


def circular_pad(x, pad):
    '''
    # x[N, 1, W, H] -> x[N, 1, W + 2 pad, H + 2 pad] (pariodic padding)
    '''
    x = torch.cat([x, x[:, :, 0:pad, :]], dim=2)
    x = torch.cat([x, x[:, :, :, 0:pad]], dim=3)
    x = torch.cat([x[:, :, -2 * pad:-pad, :], x], dim=2)
    x = torch.cat([x[:, :, :, -2 * pad:-pad], x], dim=3)
    return x


def pad_circular(input, padding):
    # type: (Tensor, List[int]) -> Tensor
    """
    Arguments
    :param input: tensor of shape :math:`(N, C_{\text{in}}, H, [W, D]))`
    :param padding: (tuple): m-elem tuple where m is the degree of convolution
    Returns
    :return: tensor of shape :math:`(N, C_{\text{in}}, [D + 2 * padding[0],
                                     H + 2 * padding[1]], W + 2 * padding[2]))`
    """
    offset = 3
    for dimension in range(input.dim() - offset + 1):
        input = dim_pad_circular(input, padding[dimension], dimension + offset)
    return input


def dim_pad_circular(input, padding, dimension):
    # type: (Tensor, int, int) -> Tensor
    input = torch.cat([input, input[[slice(None)] * (dimension - 1) +
                      [slice(0, padding)]]], dim=dimension - 1)
    input = torch.cat([input[[slice(None)] * (dimension - 1) +
                      [slice(-2 * padding, -padding)]], input], dim=dimension - 1)
    return input


def imfilter(x, k):
    '''
    x: image, NxcxHxW
    k: kernel, cx1xhxw
    '''
    x = pad_circular(x, padding=((k.shape[-2]-1)//2, (k.shape[-1]-1)//2))
    x = torch.nn.functional.conv2d(x, k, groups=x.shape[1])
    return x


def G(x, k, sf=3):
    '''
    x: image, NxcxHxW
    k: kernel, cx1xhxw
    sf: scale factor
    center: the first one or the moddle one

    Matlab function:
    tmp = imfilter(x,h,'circular');
    y = downsample2(tmp,K);
    '''
    x = downsample(imfilter(x, k), sf=sf)
    return x


def Gt(x, k, sf=3):
    '''
    x: image, NxcxHxW
    k: kernel, cx1xhxw
    sf: scale factor
    center: the first one or the moddle one

    Matlab function:
    tmp = upsample2(x,K);
    y = imfilter(tmp,h,'circular');
    '''
    x = imfilter(upsample(x, sf=sf), k)
    return x


def interpolation_down(x, sf, center=False):
    mask = torch.zeros_like(x)
    if center:
        start = torch.tensor((sf-1)//2)
        mask[..., start::sf, start::sf] = torch.tensor(1).type_as(x)
        LR = x[..., start::sf, start::sf]
    else:
        mask[..., ::sf, ::sf] = torch.tensor(1).type_as(x)
        LR = x[..., ::sf, ::sf]
    y = x.mul(mask)

    return LR, y, mask


"""
# --------------------------------------------
# degradation models
# --------------------------------------------
"""


def csmri_degradation(x, M):
    '''
    Args:
        x: 1x1xWxH image, [0, 1]
        M: mask, WxH
        n: noise, WxHx2
    '''
    x = rfft(x).mul(M.unsqueeze(-1).unsqueeze(0).unsqueeze(0)) # + n.unsqueeze(0).unsqueeze(0)
    return x


if __name__ == '__main__':

    weight = torch.tensor([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.],[7.,8.,9.],[7.,8.,9.]]).view(1,1,5,3)
    input = torch.linspace(1,9,9).view(1,1,3,3)
    input = pad_circular(input, (2,1))



