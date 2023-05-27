import numpy as np
import cv2


def down_size(img, ratio, inter_type=cv2.INTER_CUBIC):
    to_shape = (int(img.shape[1] / ratio), int(img.shape[0] / ratio))
    return cv2.resize(img, to_shape, interpolation=inter_type)

