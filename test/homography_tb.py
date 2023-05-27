import unittest

import numpy as np
import cv2

from gscan.core.geometry import homography, projection
from gscan.core.basic.regulator import GrayCuttingRegulator


class homography_tb(unittest.TestCase):
    def ignore_test_get_homography_matrix(self):
        p1 = np.array([[0, 400, 400, 0],
                       [0, 0, 300, 300]])
        p2 = np.array([[315, 696, 722, 311],
                       [234, 212, 463, 493]])

        H = homography.get_homography_matrix(p1, p2)
        print(H)

        cv2.destroyAllWindows()

    def ignore_test_homography_correction(self):
        p1 = np.array([[0, 400, 400, 0],
                       [0, 0, 360, 360]])
        p2 = np.array([[315, 696, 722, 311],
                       [234, 212, 463, 493]])
        K = np.array([[1536.1, 0, 959.5], [0, 1535.7, 723.8], [0, 0, 1]])

        img = cv2.imread("../res/test_1.jpg", cv2.IMREAD_COLOR)
        cv2.imshow("raw", img)
        cv2.waitKey(0)

        H = homography.get_homography_matrix(p1, p2)
        res = homography.homography_correction(img, H, (800, 800), regulator=GrayCuttingRegulator)

        cv2.imshow("corrected", res)
        cv2.waitKey(0)

        cv2.destroyAllWindows()

    def test_homography_correction_autoscale(self):
        # mouse control
        picked = [0]
        pixel_x = []
        pixel_y = []

        def mouse_handler(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and picked[0] < 4:
                picked[0] += 1
                pixel_x.append(x)
                pixel_y.append(y)
                cv2.circle(img_show, (x, y), 2, (0, 0, 255), 2)
                if picked[0] > 1:
                    cv2.line(img_show, (pixel_x[picked[0] - 2], pixel_y[picked[0] - 2]), (x, y), (255, 0, 0), 2)
                if picked[0] == 4:
                    cv2.line(img_show, (x, y), (pixel_x[0], pixel_y[0]), (255, 0, 0), 2)
                cv2.imshow("selecting", img_show)

        # parameters
        K = np.array([[1536.1, 0, 959.5], [0, 1535.7, 723.8], [0, 0, 1]])
        target_height = 500

        # load image
        img_raw = cv2.imread("../res/test_3.jpg", cv2.IMREAD_COLOR)

        # selecting
        img_show = np.array(img_raw)
        cv2.namedWindow("selecting")
        cv2.imshow("selecting", img_show)
        cv2.setMouseCallback("selecting", mouse_handler)
        while picked[0] < 4:
            cv2.waitKey(1)
        p_uv = np.array([pixel_x, pixel_y])

        c1, c2, c3, c4 = p_uv.swapaxes(0, 1).tolist()
        cv2.line(img_show, c1, c2, (0, 0, 255), 2)
        cv2.line(img_show, c2, c3, (0, 0, 255), 2)
        cv2.line(img_show, c3, c4, (0, 0, 255), 2)
        cv2.line(img_show, c4, c1, (0, 0, 255), 2)

        # scale calculation
        ratio = projection.calc_real_rect_ratio(K, p_uv)
        target_width = int(target_height * ratio)
        p_target = np.array([[0, target_width, target_width, 0], [0, 0, target_height, target_height]])

        # homography correction
        H = homography.get_homography_matrix(p_target, p_uv)
        res = homography.homography_correction(img_raw, H, (target_height, target_width), regulator=GrayCuttingRegulator)
        cv2.imshow("corrected", res)
        cv2.waitKey(0)

        # Example Enhancement
        res[res < 128] = 0
        res[res > 128] = 255
        cv2.imshow("enhanced", res)
        cv2.waitKey(0)

        cv2.destroyAllWindows()


if __name__ == '__main__':
    unittest.main()

