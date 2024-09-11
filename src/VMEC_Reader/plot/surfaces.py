from typing import List, Union

import matplotlib.pyplot as plt
from matplotlib import gridspec

import numpy as np

from ..transformations import FourierArray
from ..utils import check_if_iterable


def plot_RZ_surfaces(
    R: Union[FourierArray, List[FourierArray]],
    Z: Union[FourierArray, List[FourierArray]],
    surfaces: Union[int, np.ndarray, List] = 10,
    phi_angles: Union[float, np.ndarray, List] = 0.0,
    labels: str = None,
    magnetic_axis: bool = True,
    num_theta: int = 100,
    suptitle: List[str] = "Magnetic surfaces",
    phi_titles=None,
    fig_kwargs: dict = {},
    plt_kwargs: dict = {"marker": None, "linewidth": 1},
    mode="3D",
    shared_axis=True,
    show=True,
):
    """
    Plot the R - Z surfaces of a VMEC equilibrium

    Arguments
    ---------
        R: FourierArray or List of FourierArrays
            R Fourier coefficients of the surfaces
        Z: FourierArray or List of FourierArrays
            Z Fourier coefficients of the surfaces
        surfaces: int or Iterable
            if an int is given, the it will plot a linear distribution of the surfaces
            if it is an iterable, then it will take the indices in the list and plot the
            corresponding surfaces
        phi_angles: float or Iterable
            at which phi angles the surface must be represented
            if only a float, plots only one angle
        labels: str or None
            if supplied, list of labels for the list of R and Z (for comparing plots)
        magnetic_axis: bool
            plot the magnetic axis at the current angle
        num_theta: int
            number of theta points
        suptitle: str or tuple containing str and dict of arguments
            title of the figure or title with arguments
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

    R_list = R
    Z_list = Z

    colors = ["k", "r", "b", "g", "m", "c", "y"]

    if isinstance(R_list, list) or isinstance(Z_list, list):
        if not isinstance(R_list, list) or not isinstance(Z_list, list):
            raise ValueError("If one R or Z is supplied as a list, both must be lists")
        if len(R_list) != len(Z_list):
            raise ValueError("R and Z lists must have the same length")
        if labels is not None:
            if len(labels) != len(R_list):
                raise ValueError("Label list must have the same length as R and Z")
        if isinstance(plt_kwargs, list) and len(plt_kwargs) != len(R_list):
            raise ValueError("plt_kwargs list must have the same length as R and Z")
    else:
        R_list = [R]
        Z_list = [Z]

    if labels is None:
        labels = [""] * len(R_list)
    
    if isinstance(plt_kwargs, dict):
        tmp = []
        for i in range(len(R_list)):
            kwargs = plt_kwargs.copy()
            if "c" not in kwargs:
                kwargs["c"] = colors[i]
            tmp.append(kwargs)
        plt_kwargs = tmp

    phi_angles = np.array(phi_angles)


    theta = np.linspace(0, 2 * np.pi, num_theta)

    fig = plt.figure(**fig_kwargs)
    if isinstance(suptitle, str):
        fig.suptitle(suptitle)
    else:
        fig.suptitle(suptitle[0], **suptitle[1])

    gs = gridspec.GridSpec(xlen, ylen)
    axs = [plt.subplot(gs[i]) for i in range(num_plots)]
    plot_lines = [None] * len(labels)

    for i, (R, Z) in enumerate(zip(R_list, Z_list)):
        surfaces_idx = surfaces
        if isinstance(surfaces, int):
            if R.shape[0] == 1:
                surfaces_idx = None
            elif surfaces > R.shape[0]:
                surfaces_idx = np.linspace(0, R.shape[0] - 1, R.shape[0] - 1).astype(int)
            else:
                surfaces_idx = np.linspace(0, R.shape[0] - 1, surfaces).astype(int)

        print(surfaces_idx, labels[i])

        Rs = R[surfaces_idx](theta, phi_angles, mode=mode)
        Zs = Z[surfaces_idx](theta, phi_angles, mode=mode)


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

        for j, phi in enumerate(phi_angles):
            ax = axs[j]
            if phi_titles:
                ax.set_title(phi_titles[j])
            else:
                ax.set_title(rf"$\phi = {np.round(phi, 3)}$")
            if shared_axis and len(phi_angles) > 1:
                if len(phi_angles) == 2:
                    if j == 0:
                        ax.set_ylabel(r"$Z$ (m)")
                    if j == 1:
                        ax.set_yticks([])
                    ax.set_xlabel(r"$R$ (m)")
                else:
                    if j // 2 == 1:
                        ax.set_xlabel(r"$R$ (m)")
                    if j % 2 == 0:
                        ax.set_ylabel(r"$Z$ (m)")
                    if j < 2:
                        ax.set_xticks([])
            else:
                ax.set_xlabel(r"$R$ (m)")
                ax.set_ylabel(r"$Z$ (m)")
            ax.set_ylim(Zmin, Zmax)
            ax.set_xlim(Rmin, Rmax)
            ax.set_aspect("equal")

            if R.shape[0] == 1:
                plot_lines[i] = ax.plot(Rs[:, j], Zs[:, j], label=labels[i], **plt_kwargs[i])[0]
            else:
                for s in range(len(surfaces_idx)):
                    plot_lines[i] = ax.plot(Rs[s, :, j], Zs[s, :, j], label=labels[i], **plt_kwargs[i])[0]

    if labels[0]:
        fig.legend(plot_lines, labels, loc="center")

    if show:
        plt.show()
    return fig
