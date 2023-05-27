import math
import unittest
import itertools

import numpy as np
import cv2

from gscan.core.geometry import slicing, projection, homography
from gscan.core.basic import interpolation, spatial, regulator


class linear_tb(unittest.TestCase):
    def ignore_test_rt(self):
        camera = cv2.VideoCapture(0)
        while True:
            ret, frame = camera.read()

            img_down = interpolation.down_size(frame, 1)

            img_morph = spatial.morphology_opt(img_down)

            img_clane_yuv = cv2.cvtColor(img_morph, cv2.COLOR_BGR2YUV)
            img_clane_yuv[:, :, 0] = cv2.createCLAHE(3.0, (3, 3)).apply(img_clane_yuv[:, :, 0])
            img_clane = cv2.cvtColor(img_clane_yuv, cv2.COLOR_YUV2BGR)

            img_edge = slicing.edge_detection(img_clane, (40, 60))

            cv2.imshow('img', img_edge)
            cv2.waitKey(10)

    def test_edge_detecting(self):
        img_raw = cv2.imread("../res/camera_calibration/5554283BC11B6771D08774605EE59073.jpg", cv2.IMREAD_COLOR)

        img_down = interpolation.down_size(img_raw, 1)
        cv2.imshow("img", img_down)
        cv2.waitKey(0)

        img_morph = spatial.morphology_opt(img_down, (5, 5))
        # cv2.imshow("img", img_morph)
        # cv2.waitKey(0)

            # img_clane_yuv = cv2.cvtColor(img_morph, cv2.COLOR_BGR2YUV)
            # img_clane_yuv[:, :, 0] = cv2.createCLAHE(0.5, (3, 3)).apply(img_clane_yuv[:, :, 0])
            # img_clane = cv2.cvtColor(img_clane_yuv, cv2.COLOR_YUV2BGR)
            # cv2.imshow("img", img_clane)
            # cv2.waitKey(0)

        b_mean = np.mean(img_morph[:, :, 0])
        g_mean = np.mean(img_morph[:, :, 1])
        r_mean = np.mean(img_morph[:, :, 2])
        if b_mean <= g_mean and b_mean <= r_mean:
            img_mono = img_morph[:, :, 0]
        elif g_mean <= b_mean and g_mean <= r_mean:
            img_mono = img_morph[:, :, 1]
        else:
            img_mono = img_morph[:, :, 2]
        # cv2.imshow("img", img_mono)
        # cv2.waitKey(0)

        img_edge = slicing.edge_detection(img_mono, (40, 60))
        # cv2.imshow("img", img_edge)
        # cv2.waitKey(0)

        minLen = int((img_raw.shape[0] + img_raw.shape[1]) / 2 / 120)
        maxGap = int((img_raw.shape[0] + img_raw.shape[1]) / 2 / 100)
        lines = cv2.HoughLinesP(img_edge, 1, np.pi / 90, minLen, minLineLength=minLen, maxLineGap=maxGap)
        img_lines = np.zeros(img_down.shape, dtype=np.uint8)
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # cv2.imshow("img", img_lines)
        # cv2.waitKey(0)

        lines_merged = lines
        for i in range(5):
            lines_merged = slicing.merge_lines(lines_merged, 8 * math.pi / 180, 20, 100)
        lines_merged = sorted(lines_merged, key=lambda line: np.linalg.norm(line[0][:2] - line[0][2:]), reverse=True)[:20]
        img_lines_merged = np.zeros(img_down.shape, dtype=np.uint8)
        for line_merged in lines_merged:
            for x1, y1, x2, y2 in line_merged:
                cv2.line(img_lines_merged, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.imshow("img", img_lines_merged)
        cv2.waitKey(0)

        def calculate_intersection(line1, line2):
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if d != 0:
                x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
                y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
                return x, y
            else:
                return None

        def calculate_score(vertices):
            # 这里可以根据您的需求定义评分函数，评估四边形的特征，比如边长比例、角度等
            # 这里只是一个示例，简单计算四边形的面积作为评分
            x1, y1, x2, y2, x3, y3, x4, y4 = vertices
            area = 0.5 * abs((x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1) - (x2 * y1 + x3 * y2 + x4 * y3 + x1 * y4))
            return area

        def order_vertices(vertices):
            centroid_x = sum(x for x, y in vertices) / len(vertices)
            centroid_y = sum(y for x, y in vertices) / len(vertices)
            ordered_vertices = sorted(vertices, key=lambda p: (math.atan2(p[1] - centroid_y, p[0] - centroid_x) + 2 * math.pi) % (2 * math.pi))
            return ordered_vertices

        def find_best_quadrilateral(lines):
            best_score = 0
            best_quadrilateral = None

            # 遍历所有四条直线的组合
            line_combinations = itertools.combinations(lines, 4)
            for combination in line_combinations:
                # 计算四条直线的交点
                intersections = []
                line_pairs = itertools.combinations(combination, 2)
                for pair in line_pairs:
                    intersection = calculate_intersection(pair[0][0], pair[1][0])
                    if intersection and 0 <= intersection[0] < img_down.shape[1] and 0 <= intersection[1] < img_down.shape[0]:
                        intersections.append(intersection)

                # 如果有四个交点，构成四边形
                if len(intersections) >= 4:
                    centroid_x = sum(x for x, y in intersections) / 6
                    centroid_y = sum(y for x, y in intersections) / 6
                    intersections = sorted(intersections, key=lambda p: ((p[0] - centroid_x) ** 2 + (p[1] - centroid_y) ** 2))
                    points = order_vertices(intersections[:4])
                    vertices = [points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1], points[3][0], points[3][1]]

                    # 更新评分和对应的四边形
                    score = calculate_score(vertices)
                    if score > best_score:
                        best_score = score
                        best_quadrilateral = vertices

            return best_quadrilateral

        best_quadrilateral = find_best_quadrilateral(lines_merged)
        if best_quadrilateral is not None:
            p3 = (int(best_quadrilateral[0]), int(best_quadrilateral[1]))
            p4 = (int(best_quadrilateral[2]), int(best_quadrilateral[3]))
            p1 = (int(best_quadrilateral[4]), int(best_quadrilateral[5]))
            p2 = (int(best_quadrilateral[6]), int(best_quadrilateral[7]))

            img_quad = np.copy(img_down)
            cv2.line(img_quad, p1, p2, (0, 255, 0), 2)
            cv2.line(img_quad, p2, p3, (0, 255, 0), 2)
            cv2.line(img_quad, p3, p4, (0, 255, 0), 2)
            cv2.line(img_quad, p4, p1, (0, 255, 0), 2)
            cv2.imshow("img", img_quad)
            cv2.waitKey(0)

            K = np.array([[1536.1, 0, 959.5], [0, 1535.7, 723.8], [0, 0, 1]])
            target_height = 700
            p_uv = np.array([p1, p2, p3, p4]).swapaxes(0, 1)

            ratio = projection.calc_real_rect_ratio(K, p_uv)
            target_width = int(target_height * ratio)
            p_target = np.array([[0, target_width, target_width, 0], [0, 0, target_height, target_height]])

            H = homography.get_homography_matrix(p_target, p_uv)
            img_corrected = homography.homography_correction(img_down, H, (target_height, target_width), regulator=regulator.GrayCuttingRegulator)
            cv2.imshow("img", img_corrected)
            cv2.waitKey(0)

            cv2.destroyAllWindows()

if __name__ == '__main__':
    unittest.main()

