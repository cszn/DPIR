# -*- coding: utf-8 -*-
import numpy as np


'''
modified by Kai Zhang (github: https://github.com/cszn)
03/03/2019
'''


# --------------------------------
# get rho and sigma
# --------------------------------
#def get_rho_sigma(sigma=2.55/255, iter_num=15, modelSigma1=49.0, modelSigma2=2.55):
#    '''
#    One can change the sigma to implicitly change the trade-off parameter
#    between fidelity term and prior term
#    '''
#    modelSigmaS = np.logspace(np.log10(modelSigma1), np.log10(modelSigma2), iter_num).astype(np.float32)
#    sigmas = modelSigmaS/255.
#    rhos = list(map(lambda x: 0.23*(sigma**2)/(x**2), sigmas))
#    return rhos, sigmas

# --------------------------------
# get rho and sigma
# --------------------------------
def get_rho_sigma(sigma=2.55/255, iter_num=15, modelSigma1=49.0, modelSigma2=2.55, w=1.0):
    '''
    One can change the sigma to implicitly change the trade-off parameter
    between fidelity term and prior term
    '''
    modelSigmaS = np.logspace(np.log10(modelSigma1), np.log10(modelSigma2), iter_num).astype(np.float32)
    modelSigmaS_lin = np.linspace(modelSigma1, modelSigma2, iter_num).astype(np.float32)
    sigmas = (modelSigmaS*w+modelSigmaS_lin*(1-w))/255.
    rhos = list(map(lambda x: 0.23*(sigma**2)/(x**2), sigmas))
    return rhos, sigmas


def get_rho_sigma1(sigma=2.55/255, iter_num=15, modelSigma1=49.0, modelSigma2=2.55, lamda=3.0):
    '''
    One can change the sigma to implicitly change the trade-off parameter
    between fidelity term and prior term
    '''
    modelSigmaS = np.logspace(np.log10(modelSigma1), np.log10(modelSigma2), iter_num).astype(np.float32)
    sigmas = modelSigmaS/255.
    rhos = list(map(lambda x: (sigma**2)/(x**2)/lamda, sigmas))
    return rhos, sigmas


if __name__ == '__main__':
    rhos, sigmas = get_rho_sigma(sigma=2.55/255, iter_num=30, modelSigma2=2.55)
    print(rhos)
    print(sigmas*255)
