import numpy as np
import cv2


def binarization(img, ths):
    res = np.copy(img)
    res[res < ths] = 0
    res[res > ths] = 255
    return res

