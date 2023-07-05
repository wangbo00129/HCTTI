'''
Modified from torchstain
'''
import cupy as cp

def csplit(I):
    return [I[..., i] for i in range(I.shape[-1])]

def cmerge(I1, I2, I3):
    return cp.stack([I1, I2, I3], axis=-1)

def lab_split(I):
    I = I.astype("float32")
    I1, I2, I3 = csplit(I)
    return I1 / 2.55, I2 - 128, I3 - 128

def lab_merge(I1, I2, I3):
    return cmerge(I1 * 2.55, I2 + 128, I3 + 128)