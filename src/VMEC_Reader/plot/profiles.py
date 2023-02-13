from typing import List, Union

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

from ..input import InputFile


def profiles(input_file: InputFile, names=["pressure", "safety"]):
    """
    Plots the different input profiles of an input file

    Arguments
    ---------
        input_file: InputFile
        names: names of the profiles to be plotted
    """
