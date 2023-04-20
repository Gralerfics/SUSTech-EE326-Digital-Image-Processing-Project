import unittest

import numpy as np
import cv2

from gscan.core.geometry import slicing


class linear_tb(unittest.TestCase):
    def test_convolution(self):
        img_raw = cv2.imread("../res/test_1.jpg", cv2.IMREAD_COLOR)
        cv2.imshow("raw", img_raw)
        cv2.waitKey(0)

        edges = slicing.edge_detection(img_raw, (30, 80))
        cv2.imshow("smoothed", edges)
        cv2.waitKey(0)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    unittest.main()

