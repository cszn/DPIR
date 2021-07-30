# -*- coding: utf-8 -*-
import torch.fft
import torch


def splits(a, sf):
    '''split a into sfxsf distinct blocks
    Args:
        a: NxCxWxH
        sf: split factor
    Returns:
        b: NxCx(W/sf)x(H/sf)x(sf^2)
    '''
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=4)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=4)
    return b


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
    otf = torch.fft.fftn(otf, dim=(-2,-1))
    #n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    #otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
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
    FR = FBFy + torch.fft.fftn(alpha*x, dim=(-2,-1))
    x1 = FB.mul(FR)
    FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)
    invW = torch.mean(splits(F2B, sf), dim=-1, keepdim=False)
    invWBR = FBR.div(invW + alpha)
    FCBinvWBR = FBC*invWBR.repeat(1, 1, sf, sf)
    FX = (FR-FCBinvWBR)/alpha
    Xest = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))

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
    FBC = torch.conj(FB)
    F2B = torch.pow(torch.abs(FB), 2)
    STy = upsample(x, sf=sf)
    FBFy = FBC*torch.fft.fftn(STy, dim=(-2, -1))
    return FB, FBC, F2B, FBFy















