from typing import List, Union

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

from ..input import InputFile, Profile


def plot_profile(profile: Profile, ax=None, **kwargs):
    """
    Plots a profile

    Arguments
    ---------
        profile: Profile
        ax: axis to plot the profile on (if None, plt.plot is used)
        **kwargs: keyword arguments to pass to the plot
    """
    xs, ys = profile.get_xy()


    plt_fct = plt
    if not isinstance(ax, type(None)):
        plt_fct = ax

    plt_fct.plot(xs, ys, **kwargs)

    xlabel = ""

    if profile.radial_coordinate == "rho":
        xlabel = r"$\rho = \sqrt{\Psi / \Psi_b}$"
    else:
        xlabel = r"$\rho^2 = \Psi / \Psi_b$"

    ylabel = rf"{profile.name} ({profile.unit})"
    if profile.unit == "":
        ylabel = profile.name

    if ax is None:
        plt_fct.xlabel(xlabel)
        plt_fct.ylabel(ylabel)
        plt_fct.title(profile.name)
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(profile.name)


# def plot_pressure(input_file: InputFile):
#     pass


# def plot_safety(input_file: InputFile):
#     pass


# def plot_rotation(input_file: InputFile):
#     pass


# def plot_temperature(input_file: InputFile):
#     pass


def plot_profiles(input_file: InputFile, names=["pressure", "safety", "rotation", "temperature"]):
    """
    Plots the different input profiles of an input file

    Arguments
    ---------
        input_file: InputFile
        names: names of the profiles to be plotted
    """

    # plt.figure(figsize=(12, 8))

    # if "pressure" in names:
    #     plot_pressure(input_file)
    # if "safety" in names:
    #     plot_safety(input_file)
    # if "rotation" in names:
    #     plot_rotation(input_file)
    # if "temperature" in names:
    #     plot_temperature(input_file)

    # plt.show()
