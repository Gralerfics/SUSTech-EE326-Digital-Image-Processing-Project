import unittest

import glob

from gscan.tool.camera_calibration import calibration


class camera_calibration_tb(unittest.TestCase):
    def test_calibration(self):
        images = glob.glob("../res/camera_calibration/*.jpg")
        K, d, diff = calibration(images, 6, 8, display=True)
        print(K)
        print(d)
        print(diff)


if __name__ == '__main__':
    unittest.main()

