import numpy as np
import cv2


def edge_detection(img, ths=(50, 150), sobel_size=3, grad_mode='L2'):
    # [ Description ]
    #       Canny edge detecting algorithm
    # [ Arguments ]
    #       img:        the raw image
    #       ths:        the low and high thresholds
    #       sobel_size: the size of the sobel operator
    #       grad_mode:  the method to calculate the gradient (L1 / L2)
    # [ Return ]
    #       The processed image

    return cv2.Canny(img, ths[0], ths[1], apertureSize=sobel_size, L2gradient=(grad_mode == 'L2'))

