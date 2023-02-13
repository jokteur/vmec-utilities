from typing import List, Union

import matplotlib.pyplot as plt
from matplotlib import gridspec

import numpy as np

from ..transformations import FourierArray
from ..utils import check_if_iterable


def RZ_surfaces(
    R: FourierArray,
    Z: FourierArray,
    surfaces: Union[int, np.ndarray, List] = 10,
    phi_angles: Union[float, np.ndarray, List] = 0.0,
    magnetic_axis: bool = True,
    num_theta: int = 100,
    suptitle: str = "Magnetic surfaces",
    fig_kwargs: dict = {},
    plt_kwargs: dict = {"marker": None, "c": "k", "linewidth": 1},
    mode="3D",
    show=True,
):
    """
    Plot the R - Z surfaces of a VMEC equilibrium

    Arguments
    ---------
        R: FourierArray
            R Fourier coefficients of the surfaces
        Z: FourierArray
            Z Fourier coefficients of the surfaces
        surfaces: int or Iterable
            if an int is given, the it will plot a linear distribution of the surfaces
            if it is an iterable, then it will take the indices in the list and plot the
            corresponding surfaces
        phi_angles: float or Iterable
            at which phi angles the surface must be represented
            if only a float, plots only one angle
        magnetic_axis: bool
            plot the magnetic axis at the current angle
        num_theta: int
            number of theta points
        fig_kwargs: dict
            arguments for the matplotlib figure() function
        plt_kwargs: dict
            arguments for the matplotlib plot() function
        show: bool
            calls the show functions

    Returns
    -------
        corresponding figure
    """

    num_plots = 1
    xlen, ylen = 1, 1
    if check_if_iterable(phi_angles):
        num_plots = len(phi_angles)

        if num_plots == 2:
            ylen = 2
        elif num_plots <= 4:
            xlen = 2
            ylen = 2
        elif num_plots <= 6:
            xlen = 3
            ylen = 2
        elif num_plots <= 9:
            xlen = 3
            ylen = 3
        else:
            raise ValueError("To many angles specified")
    else:
        phi_angles = [phi_angles]

    phi_angles = np.array(phi_angles)

    if isinstance(surfaces, int):
        surfaces = np.linspace(0, R.shape[0] - 1, surfaces).astype(int)

    theta = np.linspace(0, 2 * np.pi, num_theta)

    Rs = R[surfaces](theta, phi_angles, mode=mode)
    Zs = Z[surfaces](theta, phi_angles, mode=mode)

    if len(phi_angles) == 1:
        Rs = Rs[..., np.newaxis]
        Zs = Zs[..., np.newaxis]

    Rmin, Rmax = np.min(Rs), np.max(Rs)
    Zmin, Zmax = np.min(Zs), np.max(Zs)

    margin_R = (Rmax - Rmin) / 10
    margin_Z = (Zmax - Zmin) / 10
    Rmin -= margin_R
    Rmax += margin_R
    Zmin -= margin_Z
    Zmax += margin_Z

    fig = plt.figure(**fig_kwargs)
    fig.suptitle(suptitle)

    gs = gridspec.GridSpec(xlen, ylen)
    axs = [plt.subplot(gs[i]) for i in range(num_plots)]

    for i, phi in enumerate(phi_angles):
        ax = axs[i]
        ax.set_title(rf"$\phi = {np.round(phi, 3)}$")
        ax.set_xlabel(r"$R$ (m)")
        ax.set_ylabel(r"$Z$ (m)")
        ax.set_ylim(Zmin, Zmax)
        ax.set_xlim(Rmin, Rmax)
        ax.set_aspect("equal")

        for s in range(len(surfaces)):
            ax.plot(Rs[s, :, i], Zs[s, :, i], **plt_kwargs)

    if show:
        plt.show()
    return fig
