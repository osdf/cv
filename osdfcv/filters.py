"""Some handy filters.
"""


import numpy as np


def binomial(width):
    """Binomial filter of width _width_.
    """
    assert width >= 2, "Binomaial Filters are only defined for widths >= 2"
    filt = np.array([0.5, 0.5])

    for i in xrange(width-2):
        filt = np.convolve(filt, [0.5, 0.5])
    return filt

def gaussian(width, sigma):
    """Gaussian filter of total width _width_ and sigma _sigma_.
    Produces equivalent output to matlab's fspecial('gaussian, width, sigma).
    """
    d, r = divmod(width, 2)
    r = 1 if r == 0 else 0
    const = -2*sigma**2
    grid = np.arange(-d + 0.5*r, d+1 - 0.5*r, 1)
    grid = np.exp((grid**2)/const)
    tmp = np.sum(grid)
    return grid/tmp

