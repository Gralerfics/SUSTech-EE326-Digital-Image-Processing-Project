import numpy as np
from ..basic.regulator import NonRegulator


def get_homography_matrix(p_a, p_b):
    # [ Description ]
    #       Calculate the homography matrix H such that p_b =(in scale)= H x p_a
    # [ Arguments ]
    #       p_a:        p_a (the iterative coords)
    #           - format: [[u0, u1, ...], [v0, v1, ...]]
    #       p_b:        p_b (the predictive coords)
    #           - format: [[u0, u1, ...], [v0, v1, ...]]
    # [ Return ]
    #       The H matrix

    pts = p_a.shape[1]
    assert pts >= 4

    A = []
    for i in range(pts):
        A.append([p_a[0][i], p_a[1][i], 1, 0, 0, 0, -p_a[0][i] * p_b[0][i], -p_a[1][i] * p_b[0][i], -p_b[0][i]])
        A.append([0, 0, 0, p_a[0][i], p_a[1][i], 1, -p_a[0][i] * p_b[1][i], -p_a[1][i] * p_b[1][i], -p_b[1][i]])
    A = np.array(A)
    h = np.linalg.lstsq(A[:, :-1], -A[:, -1], rcond=None)[0]
    return np.append(h, 1).reshape(3, 3)


def homography_correction(img_ref, H, dim, regulator=NonRegulator):
    # [ Description ]
    #       Geometry correction using H matrix
    # [ Arguments ]
    #       img_ref:    the raw image
    #       H:          the homography matrix
    #       dim:        the target viewport
    #           - format: (row, col)
    #       regulator:  The method to process the result data type
    # [ Return ]
    #       The corrected image

    mesh = np.array(np.meshgrid(range(dim[1]), range(dim[0]))).swapaxes(0, 2).swapaxes(0, 1)    # generate the mesh, coord format: (x, y)
    mesh = np.pad(mesh, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=1)                # turn into homogeneous coordinates: (x, y, 1)

    mapped = np.matmul(mesh, np.array([np.transpose(H)]))   # apply the H matrix
    mapped /= mapped[:, :, -1:]                             # normalization
    mapped = np.round(mapped[:, :, :-1]).astype(np.int32)   # turn back into ordinary, integer coordinates

    res = np.zeros((dim[0], dim[1], 3))         # (row, cols)
    flag = (mapped[:, :, 0] >= 0) & (mapped[:, :, 0] < img_ref.shape[1]) & (mapped[:, :, 1] >= 0) & (mapped[:, :, 1] < img_ref.shape[0])    # remove the overflow points
    coord = mapped[flag].swapaxes(0, 1)
    res[flag] = img_ref[coord[1], coord[0]]

    return regulator(res)

