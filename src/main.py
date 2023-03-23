import numpy as np
import cv2

from gscan.core.geometry import homography
from gscan.core.geometry import projection
from gscan.core.basic.regulator import ImageValueRegulator


K = np.array([[1536.1, 0, 959.5], [0, 1535.7, 723.8], [0, 0, 1]])

p2 = np.array([[698, 961, 981, 740],
               [431, 455, 723, 684]])

ratio = projection.calc_real_rect_ratio(K, p2)
p1_h = 500
p1_w = int(p1_h * ratio)
p1 = np.array([[0, p1_w, p1_w, 0],
               [0, 0, p1_h, p1_h]])

c1, c2, c3, c4 = p2.swapaxes(0, 1).tolist()

img = cv2.imread("../res/test_3.jpg", cv2.IMREAD_COLOR)
img_show = np.array(img)
cv2.line(img_show, c1, c2, (0, 0, 255), 2)
cv2.line(img_show, c2, c3, (0, 0, 255), 2)
cv2.line(img_show, c3, c4, (0, 0, 255), 2)
cv2.line(img_show, c4, c1, (0, 0, 255), 2)
cv2.imshow("raw", img_show)
cv2.waitKey(0)

H = homography.get_homography_matrix(p1, p2)
res = homography.homography_correction(img, H, (p1_h, p1_w), regulator=ImageValueRegulator)

cv2.imshow("corrected", res)
cv2.waitKey(0)

cv2.destroyAllWindows()