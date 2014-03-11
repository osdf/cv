"""Laplacian Pyramids.
"""


import numpy as np
import scipy.ndimage.filters as filters
try:
    import Image as img
except:
    import PIL as img


def build(im, height, down_filt, up_filt=None):
    """
    _img_ is a 2d numpy array.
    _height_ is the depth of the pyramid.
    _down_filt_ is the filter used before downsampling.
    _up_filt_ is the filter used after upsampling.
    """
    pyr = []
    
    if up_filt is None:
        up_filt = down_filt
    
    for h in xrange(height - 1):
        low = filters.convolve1d(im, down_filt, axis=0, mode='constant', cval=0)
        low = filters.convolve1d(low, down_filt, axis=1, mode='constant', cval=0)
        low = low[::2, ::2]

        dx, dy = low.shape
        high = np.zeros((2*dx, 2*dy))
        high[::2, ::2] = low
        high = filters.convolve1d(high, up_filt, axis=1)
        high = filters.convolve1d(high, up_filt, axis=0)

        diff = im - high
        pyr.append(diff)

        im = low
    pyr.append(im)
    return pyr


def reconstruct(pyr, up_filt):
    """Reconstruct original from pyramid.
    Use code 
    """
    im = pyr[-1]
    for nxt in reversed(pyr[:-1]):
        dx, dy = im.shape
        tmp = np.zeros((2*dx, 2*dy))
        tmp[::2, ::2] = im
        tmp = filters.convolve1d(tmp, up_filt, axis=1)
        tmp = filters.convolve1d(tmp, up_filt, axis=0)
        im = tmp + nxt
    return im


def build_pil(im, height, mode=img.ANTIALIAS):
    """
    """
    pyr = []
    for h in xrange(height - 1):
        pimg = img.fromarray(im)
        dx, dy = pimg.size
        low = pimg.resize((dx/2, dy/2), resample=mode)
        high = low.resize((dx, dy), resample=mode)
        
        diff = np.asarray(pimg, dtype=np.float) - np.asarray(high, dtype=np.float)
        pyr.append(diff)

        im = np.asarray(low, dtype=np.float)
    pyr.append(im)
    return pyr


def reconstruct_pil(pyr, mode=img.ANTIALIAS):
    """
    """
    im = img.fromarray(pyr[-1])
    for nxt in reversed(pyr[:-1]):
        dx, dy = im.size
        high = im.resize((2*dx, 2*dy), resample=mode)
        tmp = nxt + np.asarray(high, dtype=np.float)
        im = img.fromarray(tmp)
    return np.asarray(im, dtype=np.float)

