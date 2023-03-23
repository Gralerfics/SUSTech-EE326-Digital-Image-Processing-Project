import numpy as np


def NonRegulator(dat):
    # [ Description ]
    #       Do-nothing regulator
    # [ Arguments ]
    #       mat:    Input data
    # [ Return ]
    #       The raw data

    return dat


def ImageValueRegulator(dat):
    # [ Description ]
    #       A regulator that clip the data between 0 and 255
    #       and convert the data into numpy.uint8
    # [ Arguments ]
    #       mat:    Input data
    # [ Return ]
    #       The regulated data

    return np.clip(dat, 0, 255).astype(np.uint8)

