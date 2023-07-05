"""
Modified from https://github.com/wanghao14/Stain_Normalization
A. Vahadane et al., ‘Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images’, IEEE Transactions on Medical Imaging, vol. 35, no. 8, pp. 1962–1971, Aug. 2016.

"""

from __future__ import division

import cupy as cp
from dictlearn_gpu.utils import dct_dict_1d
from dictlearn_gpu import train_dict

def standardize_brightness(I):
    """

    :param I:
    :return:
    """
    p = cp.percentile(I, 90)
    return cp.clip(I * 255.0 / p, 0, 255).astype(cp.uint8)

def remove_zeros(I):
    """
    Remove zeros, replace with 1's.
    :param I: uint8 array
    :return:
    """
    mask = (I == 0)
    I[mask] = 1
    return I

def RGB_to_OD(I):
    """
    Convert from RGB to optical density
    :param I:
    :return:
    """
    I = remove_zeros(I)
    return -1 * cp.log(I / 255)

def OD_to_RGB(OD):
    """
    Convert from optical density to RGB
    :param OD:
    :return:
    """
    return (255 * cp.exp(-1 * OD)).astype(cp.uint8)

def normalize_rows(A):
    """
    Normalize rows of an array
    :param A:
    :return:
    """
    return A / cp.linalg.norm(A, axis=1)[:, None]

def notwhite_mask(I, thresh=0.8):
    """
    Get a binary mask where true denotes 'not white'
    :param I:
    :param thresh:
    :return:
    """
    I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
    L = I_LAB[:, :, 0] / 255.0
    return (L < thresh)

def get_stain_matrix(I, threshold=0.8, lamda=0.1):
    """
    Get 2x3 stain matrix. First row H and second row E
    :param I:
    :param threshold:
    :param lamda:
    :return:
    """
    mask = notwhite_mask(I, thresh=threshold).reshape((-1,))
    OD = RGB_to_OD(I).reshape((-1, 3))
    OD = OD[mask]
    if OD.size == 0:
        raise ValueError("all pixels have all been masked as being to bright")
    # dictionary = spams.trainDL(OD.T, K=2, lambda1=lamda, mode=2, modeD=0, posAlpha=True, posD=True, verbose=False).T
    
    OD = cp.asarray(OD, dtype='float32')
    L = OD.T.shape[0]
    dictionary = dct_dict_1d(
        n_atoms=2,
        size=L,
    )
    dictionary, errors, iters = train_dict(OD.T, dictionary, sparsity_target=8, \
        min_error=0.001, max_iters=100, callbacks=None, verbose=0, as_gpu=True)
    
    if dictionary[0, 0] < dictionary[1, 0]:
        dictionary = dictionary[[1, 0], :]
    dictionary = normalize_rows(dictionary)
    return dictionary


###

class Normalizer(object):
    """
    A stain normalization object
    """

    def __init__(self):
        self.stain_matrix_target = None

    def fit(self, target):
        target = standardize_brightness(target)
        self.stain_matrix_target = get_stain_matrix(target)

    def target_stains(self):
        return OD_to_RGB(self.stain_matrix_target)

    def transform(self, I):
        I = ut.standardize_brightness(I)
        stain_matrix_source = get_stain_matrix(I)
        source_concentrations = ut.get_concentrations(I, stain_matrix_source)
        return (255 * cp.exp(-1 * cp.dot(source_concentrations, self.stain_matrix_target).reshape(I.shape))).astype(
            cp.uint8)

    def hematoxylin(self, I):
        I = standardize_brightness(I)
        h, w, c = I.shape
        stain_matrix_source = get_stain_matrix(I)
        source_concentrations = ut.get_concentrations(I, stain_matrix_source)
        H = source_concentrations[:, 0].reshape(h, w)
        H = cp.exp(-1 * H)
        return H
