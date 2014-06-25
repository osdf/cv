"""Foveated Pyramid
"""


import numpy as np


def build(im, height):
    """
    _img_ is a 2d numpy array.
    _height_ is the depth of the pyramid.
    """
    pyr = [im]
    cx, cy = im.shape[0]//2, im.shape[1]//2
    dx, dy = cx//2, cy//2
    for h in xrange(height-1):
        pyr.append(im[cx-dx:cx+dx, cy-dy:cy+dy])
        dx, dy = dy//2, dy//2
    return pyr
