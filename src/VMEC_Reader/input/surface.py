from .parameters import InputFile
from ..transformations import FourierArray, fourier_transform

import numpy as np

def input_file_surface(input_file: InputFile) -> FourierArray:
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
    
    xm = [t[0] for t in rbc.indices]
    xn = [t[1] for t in rbc.indices]

    return FourierArray(xm, xn, [rbc.array]), FourierArray(xm, xn, None, [zbs.array])
