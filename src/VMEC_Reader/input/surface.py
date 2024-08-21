from .parameters import InputFile
from ..transformations import FourierArray, fourier_transform
from typing import Tuple

import numpy as np

def input_file_boundary(input_file: InputFile) -> Tuple[FourierArray, FourierArray]:
    """
    Converts the input file surface guess to a FourierArray

    Arguments
    ---------
        input_file: InputFile
            input file to convert

    Returns
    -------
        FourierArray
            FourierArray representation of the R coordinate
        FourierArray
            FourierArray representation of the Z coordinate
    """
    
    try:
        input_file.rbc
        input_file.zbs
    except AttributeError:
        raise AttributeError("The input file should have rbc and zbs variables to convert to a FourierArray")
    
    rbc = input_file.rbc.data
    zbs = input_file.zbs.data

    if rbc.indices != zbs.indices:
        raise ValueError("The rbc and zbs variables should have the same mode indices")
    
    xm_R = np.array([t[0] for t in rbc.indices])
    xn_R = np.array([t[1] for t in rbc.indices])
    xm_Z = np.array([t[0] for t in zbs.indices])
    xn_Z = np.array([t[1] for t in zbs.indices])

    return FourierArray(xm_R, xn_R, rbc.array), FourierArray(xm_Z, xn_Z, None, zbs.array)
