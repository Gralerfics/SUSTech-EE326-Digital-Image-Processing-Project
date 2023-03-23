import numpy as np
import cv2

def calibration(imgs, rows = 6, cols = 8, display=False, display_interval=50):
    r = rows - 1
    c = cols - 1
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    objp = np.zeros((r * c, 3), np.float32)
    objp[:, :2] = np.mgrid[0:r, 0:c].T.reshape(-1, 2)
    obj_points = []
    img_points = []
    for fname in imgs:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, (5, 7), None)
        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            img_points.append(corners2)

            if display:
                cv2.drawChessboardCorners(img, (r, c), corners, ret)
                cv2.imshow('calibration_img', img)
                cv2.waitKey(display_interval)
    if display:
        cv2.destroyAllWindows()

    diff, K, distort_args, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    return (K, distort_args, diff)

