import numpy as np
from .regulator import NonRegulator


def convolution(img, kernel, pad_mode='reflect', regulator=NonRegulator):
    # [ Description ]
    #       Convolution on images
    # [ Arguments ]
    #       img:        Input image which is IMREAD_COLOR or IMREAD_GRAYSCALE
    #       kernel      Convolution kernel (length >= 3 and is odd)
    #       pad_mode:   The method to expand the edge of the image
    #       regulator:  The method to process the result data type
    # [ Return ]
    #       The convoluted image

    size = kernel.shape[0]
    rad = size // 2

    if len(img.shape) == 2:
        img = np.pad(img.astype(np.int32), ((rad, rad), (rad, rad)), pad_mode)
        kernel_spanned = np.expand_dims(np.reshape(kernel, -1), axis=(1, 2))
    else:
        img = np.pad(img.astype(np.int32), ((rad, rad), (rad, rad), (0, 0)), pad_mode)
        kernel_spanned = np.expand_dims(np.reshape(kernel, -1), axis=(1, 2, 3))

    pts = [(idx // size - rad, idx % size - rad) for idx in range(size * size)]
    moved = np.array([np.roll(np.roll(img, x, axis=0), y, axis=1) for (x, y) in pts])
    res = np.sum(moved * kernel_spanned, axis=0)

    return regulator(res[rad:-rad, rad:-rad])

