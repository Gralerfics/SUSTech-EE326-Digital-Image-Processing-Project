import numpy as np
import cv2

from gscan.core.geometry import homography
from gscan.core.geometry import projection
from gscan.core.basic.regulator import ImageValueRegulator


# mouse control
picked = 0
pixel_x = []
pixel_y = []

def mouse_handler(event, x, y, flags, param):
    global picked
    if event == cv2.EVENT_LBUTTONDOWN and picked < 4:
        picked += 1
        pixel_x.append(x)
        pixel_y.append(y)
        cv2.circle(img_show, (x, y), 2, (0, 0, 255), 2)
        if picked > 1:
            cv2.line(img_show, (pixel_x[picked - 2], pixel_y[picked - 2]), (x, y), (255, 0, 0), 2)
        if picked == 4:
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
while picked < 4:
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
res = homography.homography_correction(img_raw, H, (target_height, target_width), regulator=ImageValueRegulator)
cv2.imshow("corrected", res)
cv2.waitKey(0)

cv2.destroyAllWindows()