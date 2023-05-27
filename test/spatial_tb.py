import unittest

import numpy as np
import cv2

from gscan.core.basic import spatial
from gscan.core.basic.regulator import GrayCuttingRegulator


class spatial_tb(unittest.TestCase):
    def test_convolution(self):
        img = cv2.imread("../res/test_1.jpg", cv2.IMREAD_COLOR)
        cv2.imshow("raw", img)
        cv2.waitKey(0)

        kernel = np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ])
        img_smoothed = spatial.convolution(img, kernel, regulator=GrayCuttingRegulator)
        cv2.imshow("smoothed", img_smoothed)
        cv2.waitKey(0)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    unittest.main()

