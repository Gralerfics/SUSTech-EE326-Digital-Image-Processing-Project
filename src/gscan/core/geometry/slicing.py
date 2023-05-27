import math
import itertools

import numpy as np
import cv2

from ..basic import spatial


def edge_detection(img, ths=(50, 150), sobel_size=3, grad_mode='L2'):
    # [ Description ]
    #       Canny edge detecting algorithm
    # [ Arguments ]
    #       img:        the raw image
    #       ths:        the low and high thresholds
    #       sobel_size: the size of the sobel operator
    #       grad_mode:  the method to calculate the gradient (L1 / L2)
    # [ Return ]
    #       The processed image

    return cv2.Canny(img, ths[0], ths[1], apertureSize=sobel_size, L2gradient=(grad_mode == 'L2'))


def merge_lines(lines, threshold_angle, threshold_distance, threshold_parallel):
    merged_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        angle = np.arctan2(dy, dx)

        merged = False
        for merged_line in merged_lines:
            merged_x1, merged_y1, merged_x2, merged_y2 = merged_line[0]
            merged_dx = merged_x2 - merged_x1
            merged_dy = merged_y2 - merged_y1
            merged_angle = np.arctan2(merged_dy, merged_dx)

            if np.abs(angle - merged_angle) <= threshold_angle:
                alpha_1 = np.arctan2(y1 - merged_y1, x1 - merged_x1) - merged_angle
                alpha_2 = np.arctan2(y2 - merged_y1, x2 - merged_x1) - merged_angle
                dist_1 = np.sqrt((x1 - merged_x1) ** 2 + (y1 - merged_y1) ** 2)
                dist_2 = np.sqrt((x2 - merged_x1) ** 2 + (y2 - merged_y1) ** 2)
                line_distance = np.abs(np.sin(alpha_1) * dist_1)

                merged_d1 = 0
                merged_d2 = np.sqrt(merged_dx ** 2 + merged_dy ** 2)
                d1 = np.cos(alpha_1) * dist_1
                d2 = np.cos(alpha_2) * dist_2

                if (merged_d1 < d1 < merged_d2 or merged_d1 < d2 < merged_d2):
                    parallel_distance = 0
                else:
                    parallel_distance = min(
                        np.abs(merged_d1 - d1),
                        np.abs(merged_d1 - d2),
                        np.abs(merged_d2 - d1),
                        np.abs(merged_d2 - d2)
                    )

                if line_distance <= threshold_distance and parallel_distance <= threshold_parallel:
                    min_d = min(merged_d1, merged_d2, d1, d2)
                    max_d = max(merged_d1, merged_d2, d1, d2)
                    if min_d == merged_d1:
                        merged_line[0][0] = merged_x1
                        merged_line[0][1] = merged_y1
                    elif min_d == merged_d2:
                        merged_line[0][0] = merged_x2
                        merged_line[0][1] = merged_y2
                    elif min_d == d1:
                        merged_line[0][0] = x1
                        merged_line[0][1] = y1
                    else:
                        merged_line[0][0] = x2
                        merged_line[0][1] = y2
                    if max_d == merged_d1:
                        merged_line[0][2] = merged_x1
                        merged_line[0][3] = merged_y1
                    elif max_d == merged_d2:
                        merged_line[0][2] = merged_x2
                        merged_line[0][3] = merged_y2
                    elif max_d == d1:
                        merged_line[0][2] = x1
                        merged_line[0][3] = y1
                    else:
                        merged_line[0][2] = x2
                        merged_line[0][3] = y2
                    merged = True

        if not merged:
            merged_lines.append(line)

    return merged_lines


def slice_image(img):
    def intersect(line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if d != 0:
            x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
            y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
            return x, y
        else:
            return None

    def evaluate(vertices):
        x1, y1, x2, y2, x3, y3, x4, y4 = vertices
        area = 0.5 * abs((x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1) - (x2 * y1 + x3 * y2 + x4 * y3 + x1 * y4))
        return area

    def order_vertices(vertices):
        centroid_x = sum(x for x, y in vertices) / len(vertices)
        centroid_y = sum(y for x, y in vertices) / len(vertices)
        ordered_vertices = sorted(vertices, key=lambda p: (math.atan2(p[1] - centroid_y, p[0] - centroid_x) + 2 * math.pi) % (2 * math.pi))
        return ordered_vertices

    def find_best_quadrilateral(lines):
        quadrilaterals = []
        line_combinations = itertools.combinations(lines, 4)
        for combination in line_combinations:
            intersections = []
            line_pairs = itertools.combinations(combination, 2)
            for pair in line_pairs:
                intersection = intersect(pair[0][0], pair[1][0])
                if intersection and 0 <= intersection[0] < dim[1] and 0 <= intersection[1] < dim[0]:
                    intersections.append(intersection)
            if len(intersections) >= 4:
                centroid_x = sum(x for x, y in intersections) / 6
                centroid_y = sum(y for x, y in intersections) / 6
                intersections = sorted(intersections, key=lambda p: ((p[0] - centroid_x) ** 2 + (p[1] - centroid_y) ** 2))
                points = order_vertices(intersections[:4])
                vertices = [points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1], points[3][0], points[3][1]]
                score = evaluate(vertices)
                quadrilaterals.append((score, vertices))
        quadrilaterals = sorted(quadrilaterals, key=lambda q: q[0], reverse=True)
        return quadrilaterals

    dim = img.shape

    # Morphological operation
    img_morph = spatial.morphology_opt(img, (5, 5))

    # Pick the channel with the lowest mean value
    b_mean = np.mean(img_morph[:, :, 0])
    g_mean = np.mean(img_morph[:, :, 1])
    r_mean = np.mean(img_morph[:, :, 2])
    if b_mean <= g_mean and b_mean <= r_mean:
        img_mono = img_morph[:, :, 0]
    elif g_mean <= b_mean and g_mean <= r_mean:
        img_mono = img_morph[:, :, 1]
    else:
        img_mono = img_morph[:, :, 2]

    # Canny edge detection
    img_edge = edge_detection(img_mono, (40, 60))

    # Hough transform
    minLen = int((dim[0] + dim[1]) / 2 / 120)
    maxGap = int((dim[0] + dim[1]) / 2 / 100)
    lines = cv2.HoughLinesP(img_edge, 1, np.pi / 90, minLen, minLineLength=minLen, maxLineGap=maxGap)

    # Merge lines
    lines_merged = lines
    for i in range(5):
        lines_merged = merge_lines(lines_merged, 8 * math.pi / 180, 20, 100)
    lines_merged = sorted(lines_merged, key=lambda line: np.linalg.norm(line[0][:2] - line[0][2:]), reverse=True)[:20]

    # Find best quadrilateral
    quadrilaterals = find_best_quadrilateral(lines_merged)
    if quadrilaterals is not None:
        best_quadrilateral = quadrilaterals[0][1]
        p2 = (int(best_quadrilateral[0]), int(best_quadrilateral[1]))
        p3 = (int(best_quadrilateral[2]), int(best_quadrilateral[3]))
        p0 = (int(best_quadrilateral[4]), int(best_quadrilateral[5]))
        p1 = (int(best_quadrilateral[6]), int(best_quadrilateral[7]))
        return p0, p1, p2, p3
    else:
        return None

