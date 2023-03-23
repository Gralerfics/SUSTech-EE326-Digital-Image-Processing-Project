import numpy as np


def calc_z_inscale(p):
    # [ Description ]
    #       Calculate the z values (in scale) of the four corners in the real scene
    #       In fact is to calculate the null space of the matrix A:
    #           [[u1, -u2,  u3, -u4]
    #            [v1, -v2,  v3, -v4]
    #            [ 1,  -1,   1,  -1]]
    # [ Arguments ]
    #       p:          the pixel coordinates
    #           - format: [[u1, u2, u3, u4], [v1, v2, v3, v4]]
    # [ Return ]
    #       A matrix like [u, v, w, 1], which represents z1 = u * z4, z2 = v * z4, z3 = w * z4
    #       [0, 0, 0, -1] will be returned when the solution is not found

    u1, u2, u3, u4 = p[0]
    v1, v2, v3, v4 = p[1]

    fac = u1*v2 - u2*v1 - u1*v3 + u3*v1 + u2*v3 - u3*v2
    if fac == 0:
        return [0, 0, 0, -1]

    u = (u2*v3 - u3*v2 - u2*v4 + u4*v2 + u3*v4 - u4*v3) / fac
    v = (u1*v3 - u3*v1 - u1*v4 + u4*v1 + u3*v4 - u4*v3) / fac
    w = (u1*v2 - u2*v1 - u1*v4 + u4*v1 + u2*v4 - u4*v2) / fac
    if u < 0 or v < 0 or w < 0:
        return [0, 0, 0, -1]

    return np.array([u, v, w, 1])


def calc_P_inscale(K, p):
    # [ Description ]
    #       Calculate the 3-D points (in scale)
    # [ Arguments ]
    #       K:          the intrinsic matrix of the camera
    #       p:          the pixel coordinates
    #           - format: [[u1, u2, u3, u4], [v1, v2, v3, v4]]
    # [ Return ]
    #       The coordinates of the four corners in the real scene: P_1, P_2, P_3 and P_4 (in scale)

    z_scale = calc_z_inscale(p)
    K_inv = np.linalg.inv(K)
    p_homo = np.append(p, [1, 1, 1, 1]).reshape(3, -1)
    return (np.matmul(K_inv, p_homo) * z_scale).swapaxes(0, 1)


def calc_real_rect_ratio(K, p):
    # [ Description ]
    #       Calculate the ratio of the rectangle area in the real scene
    # [ Arguments ]
    #       K:          the intrinsic matrix of the camera
    #       p:          the pixel coordinates
    #           - format: [[u1, u2, u3, u4], [v1, v2, v3, v4]]
    # [ Return ]
    #       The value of l_21 / l_23 (clockwise, and 1 is the left-top point)

    P1, P2, P3, P4 = calc_P_inscale(K, p)
    return np.linalg.norm(P1 - P2) / np.linalg.norm(P3 - P2)

