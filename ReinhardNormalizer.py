'''
Modified from torchstain
'''
import cupy as cp
from ColorSpaceUtils import rgb2lab, lab2rgb
from SplitUtils import lab_split, lab_merge
from StatsUtils import get_mean_std, standardize

"""
Source code adapted from:
https://github.com/DigitalSlideArchive/HistomicsTK/blob/master/histomicstk/preprocessing/color_normalization/reinhard.py
https://github.com/Peter554/StainTools/blob/master/staintools/reinhard_color_normalizer.py
"""
class CupyReinhardNormalizer():
    def __init__(self):
        self.target_mus = None
        self.target_stds = None
    
    def fit(self, target):
        # normalize
        target = target.astype("float32") / 255

        # convert to LAB
        lab = rgb2lab(target)

        # get summary statistics
        stack_ = cp.array([get_mean_std(x) for x in lab_split(lab)])
        self.target_means = stack_[:, 0]
        self.target_stds = stack_[:, 1]

    def normalize(self, I):
        # normalize
        I = I.astype("float32") / 255
        
        # convert to LAB
        lab = rgb2lab(I)
        labs = lab_split(lab)

        # get summary statistics from LAB
        stack_ = cp.array([get_mean_std(x) for x in labs])
        mus = stack_[:, 0]
        stds = stack_[:, 1]

        # standardize intensities channel-wise and normalize using target mus and stds
        result = [standardize(x, mu_, std_) * std_T + mu_T for x, mu_, std_, mu_T, std_T \
            in zip(labs, mus, stds, self.target_means, self.target_stds)]
        
        # rebuild LAB
        lab = lab_merge(*result)

        # convert back to RGB from LAB
        lab = lab2rgb(lab)

        # rescale to [0, 255] uint8
        return (lab * 255).astype("uint8")
