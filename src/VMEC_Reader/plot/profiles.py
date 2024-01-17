from typing import List, Union

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

from ..input import InputFile


def plot_pressure(input_file: InputFile):
    pass


def plot_safety(input_file: InputFile):
    pass


def plot_rotation(input_file: InputFile):
    pass


def plot_temperature(input_file: InputFile):
    pass


def plot_profiles(
    input_file: InputFile, names=["pressure", "safety", "rotation", "temperature"]
):
    """
    Plots the different input profiles of an input file

    Arguments
    ---------
        input_file: InputFile
        names: names of the profiles to be plotted
    """

    if "pressure" in names:
        plot_pressure(input_file)
    if "safety" in names:
        plot_safety(input_file)
    if "rotation" in names:
        plot_rotation(input_file)
    if "temperature" in names:
        plot_temperature(input_file)

    plt.show()
