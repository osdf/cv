"""LCN Pyramid.
(Local contrast normalization from LeCun.)
"""


import numpy as np
import theano
import theano.tensor as T

import scipy.ndimage.filters as filters

try:
    import Image as img
except:
    import PIL as img


def gaussian2d(width, sigma):
    """
    Gaussian filter of total width _width_.
    """
    d, r = divmod(width, 2)
    r = 1 if r==0 else 0
    const = -2*sigma**2
    grid = np.arange(-d + 0.5*r, d+1 - 0.5*r, 1)
    grid = np.exp((grid**2)/const)
    gridT = grid.reshape(grid.size, 1)
    result = gridT*grid
    return np.asarray(result/np.sum(result), dtype=theano.config.floatX)


def lcn_filters(fmaps, depth, width, sigma):
    """
    """
    filters = np.zeros((fmaps, fmaps, width, width), dtype=theano.config.floatX)
    d2 = depth//2
    for i in xrange(fmaps):
        for j in xrange(i-d2, i+d2+1):
            if (j >= 0) and (j <fmaps):
                filters[i, j, :, :] = gaussian2d(width, sigma)
        fi_sum = np.sum(filters[i])
        filters[i] /= fi_sum
    return filters


def lcn_sublayer(image, image_shape, fmaps, pool_depth, width, sigma):
    """
    """
    print image_shape
    print fmaps, pool_depth, width, sigma
    border = width//2
    filters = lcn_filters(fmaps, pool_depth, width, sigma) 
    filter_shape = filters.shape
    blurred_mean = conv.conv2d(input=image, filters=filters, 
            image_shape=image_shape, filter_shape=filter_shape,
            border_mode='full')
    image -= blurred_mean[:, :, border:-border, border:-border]
    
    image_sqr = T.sqr(image)
    blurred_sqr = conv.conv2d(input=image_sqr, filters=filters, 
            image_shape=image_shape, filter_shape=filter_shape,
            border_mode='full')

    div = T.sqrt(blurred_sqr[:, :, border:-border, border:-border])
    fm_mean = div.mean(axis=[2, 3])
    div = T.largest(fm_mean.dimshuffle(0, 1, 'x', 'x'), div) + 1e-6
    image = image/div
    return T.cast(image, theano.config.floatX)


def build_pil(im, height, lcns, mode=img.ANTIALIAS):
    """
    _lcns_: The LCN filters, as many as the pyramid is deep.
    """
    pyr = []
    assert height == len(lcns), "Not enought filter steps."

    for h in xrange(height):
        lcn = lcns[h](im.reshape(1, 1, im.shape[0], im.shape[1]))
        lcn = lcn[0, 0, :, :]
        
        pyr.append(lcn)

        pimg = img.fromarray(lcn)
        dx, dy = pimg.size
        low = pimg.resize((dx/2, dy/2), resample=mode)

        im = np.asarray(low, dtype=np.float)
    return pyr
