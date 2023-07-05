'''
Modified from torchstain
'''
import cupy as cp

# constant conversion matrices between color spaces: https://gist.github.com/bikz05/6fd21c812ef6ebac66e1
_rgb2xyz = cp.array([[0.412453, 0.357580, 0.180423],
                     [0.212671, 0.715160, 0.072169],
                     [0.019334, 0.119193, 0.950227]])

"""
Implementation adapted from:
https://gist.github.com/bikz05/6fd21c812ef6ebac66e1
https://github.com/scikit-image/scikit-image/blob/00177e14097237ef20ed3141ed454bc81b308f82/skimage/color/colorconv.py#L704
"""
def rgb2lab(rgb):
    '''
    rgb should have been divided by 255. 
    '''
    rgb = rgb.astype("float32")

    # convert rgb -> xyz color domain
    arr = rgb.copy()
    mask = arr > 0.04045
    arr[mask] = cp.power((arr[mask] + 0.055) / 1.055, 2.4)
    arr[~mask] /= 12.92
    xyz = cp.dot(arr, _rgb2xyz.T.astype(arr.dtype))

    # scale by CIE XYZ tristimulus values of the reference white point
    arr = xyz.copy()
    arr = arr / cp.asarray((0.95047, 1., 1.08883), dtype=xyz.dtype)

    # Nonlinear distortion and linear transformation
    mask = arr > 0.008856
    arr[mask] = cp.cbrt(arr[mask])
    arr[~mask] = 7.787 * arr[~mask] + 16. / 116.

    x, y, z = arr[..., 0], arr[..., 1], arr[..., 2]

    # Vector scaling
    L = (116. * y) - 16.
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)

    # OpenCV format
    L *= 2.55
    a += 128
    b += 128

    # finally, get LAB color domain
    return cp.concatenate([x[..., cp.newaxis] for x in [L, a, b]], axis=-1)

_xyz2rgb = cp.linalg.inv(_rgb2xyz)

"""
Implementation is based on:
https://github.com/scikit-image/scikit-image/blob/00177e14097237ef20ed3141ed454bc81b308f82/skimage/color/colorconv.py#L704
"""
def lab2rgb(lab):
    '''
    The values returned should be multiplied by 255. 
    '''
    lab = lab.astype("float32")
    # first rescale back from OpenCV format
    lab[..., 0] /= 2.55
    lab[..., 1] -= 128
    lab[..., 2] -= 128

    # convert LAB -> XYZ color domain
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    y = (L + 16.) / 116.
    x = (a / 500.) + y
    z = y - (b / 200.)

    out = cp.stack([x, y, z], axis=-1)

    mask = out > 0.2068966
    out[mask] = cp.power(out[mask], 3.)
    out[~mask] = (out[~mask] - 16.0 / 116.) / 7.787

    # rescale to the reference white (illuminant)
    out *= cp.array((0.95047, 1., 1.08883), dtype=out.dtype)
    
    # convert XYZ -> RGB color domain
    arr = out.copy()
    arr = cp.dot(arr, _xyz2rgb.T)
    mask = arr > 0.0031308
    arr[mask] = 1.055 * cp.power(arr[mask], 1 / 2.4) - 0.055
    arr[~mask] *= 12.92
    return cp.clip(arr, 0, 1)