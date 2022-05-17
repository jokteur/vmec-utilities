"""
Author: Joachim Koerfer
"""
from audioop import tostereo
from typing import Any, List, Union
import numpy as np
from numba import njit

from .utils import compare_shapes, Numeric, check_if_iterable


@njit(cache=True, nogil=True)
def fast_cos(
    out_array: np.ndarray,
    outer_shape: tuple,
    theta: np.ndarray,
    phi: np.ndarray,
    n: np.ndarray,
    m: np.ndarray,
    C: np.ndarray,
    mode: str,
):
    if mode == "2d":
        restrict_idx = np.argwhere(n == 0)[0]
    else:
        restrict_idx = np.arange(len(m))

    for outer_idx in np.ndindex(outer_shape):
        for phi_idx in np.arange(len(phi)):
            # Theta is ellipsed, as we do vector operation on it
            idx = (*outer_idx, phi_idx)
            for i in restrict_idx:
                val = m[i] * theta - n[i] * phi[phi_idx]
                coeff_idx = (*outer_idx, i)
                out_array[idx] += C[coeff_idx] * np.cos(val)


@njit(cache=True, nogil=True)
def fast_sin(
    out_array: np.ndarray,
    outer_shape: tuple,
    theta: np.ndarray,
    phi: np.ndarray,
    n: np.ndarray,
    m: np.ndarray,
    S: np.ndarray,
    mode: str,
):
    if mode == "2d":
        restrict_idx = np.argwhere(n == 0)[0]
    else:
        restrict_idx = np.arange(len(m))

    for outer_idx in np.ndindex(outer_shape):
        for phi_idx in np.arange(len(phi)):
            # Theta is ellipsed, as we do vector operation on it
            idx = (*outer_idx, phi_idx)
            for i in restrict_idx:
                val = m[i] * theta - n[i] * phi[phi_idx]
                coeff_idx = (*outer_idx, i)
                out_array[idx] += S[coeff_idx] * np.sin(val)


@njit(cache=True, nogil=True)
def fast_sincos(
    out_array: np.ndarray,
    outer_shape: tuple,
    theta: np.ndarray,
    phi: np.ndarray,
    n: np.ndarray,
    m: np.ndarray,
    C: np.ndarray,
    S: np.ndarray,
    mode: str,
):
    if mode == "2d":
        restrict_idx = np.argwhere(n == 0)[0]
    else:
        restrict_idx = np.arange(len(m))

    for outer_idx in np.ndindex(outer_shape):
        for phi_idx in np.arange(len(phi)):
            # Theta is ellipsed, as we do vector operation on it
            idx = (*outer_idx, phi_idx)
            for i in restrict_idx:
                val = m[i] * theta - n[i] * phi[phi_idx]
                coeff_idx = (*outer_idx, i)
                out_array[idx] += C[coeff_idx] * np.cos(val) + S[coeff_idx] * np.sin(val)


class FourierArray:
    """
    An array to store and calculate values from Fourier coefficients
    """

    m_indices: np.ndarray
    n_indices: np.ndarray
    cos_coeff: np.ndarray
    sin_coeff: np.ndarray

    def __init__(
        self,
        m_indices: Union[List, np.ndarray],
        n_indices: Union[List, np.ndarray],
        cos_coeff: Union[List, np.ndarray] = None,
        sin_coeff: Union[List, np.ndarray] = None,
    ) -> None:
        """
        FourierArray can store and use n-dimensional coefficients

        It is assumed that the last dimension of the coefficients is the dimension of the coefficients.

        Arguments
        ---------
            m_indices: list of m numbers
            n_indices: list of n_numbers
            cos_coeff: cosinus Fourier harmonics
        """
        # Avoid useless copies
        if not isinstance(m_indices, np.ndarray):
            self.m_indices = np.array(m_indices)
        else:
            self.m_indices = m_indices

        if not isinstance(n_indices, np.ndarray):
            self.n_indices = np.array(n_indices)
        else:
            self.n_indices = n_indices

        if not isinstance(cos_coeff, np.ndarray):
            self.cos_coeff = np.array(cos_coeff)
        else:
            self.cos_coeff = cos_coeff

        if not isinstance(sin_coeff, np.ndarray):
            self.sin_coeff = np.array(sin_coeff)
        else:
            self.sin_coeff = sin_coeff
        self.pre_sliced = None

        if not self.cos_coeff.any() and not self.sin_coeff.any():
            raise ValueError(
                "At least one series of coefficients (either sinus or cosinus) must be given"
            )

        if not compare_shapes(self.m_indices.shape, self.n_indices.shape):
            raise ValueError("m and n indices have different shapes")

        if len(self.m_indices.shape) > 1:
            raise ValueError("m and n arrays should be one dimensional")

        # Assumes that the last dimension is the dimension of the coefficients
        if (
            self.cos_coeff.any()
            and self.cos_coeff.shape[-1] != self.m_indices.shape[0]
            or self.sin_coeff.any()
            and self.sin_coeff.shape[-1] != self.m_indices.shape[0]
        ):
            raise ValueError(
                "Size of last dimension of coefficients should be the same as the indices."
            )

    def check_slice(self, key: slice):
        # Check if slice is valid with dimensions of coefficients
        valid_array = self.cos_coeff if self.cos_coeff.any() else self.sin_coeff
        if valid_array.ndim == 1 and key != slice(None):
            raise IndexError("Cannot slice a 0 dimensional array")
        valid_array[..., 0][key]

    def __getitem__(self, key: slice) -> "FourierArray":
        """
        Slice the array before calling ()
        """
        if self.pre_sliced:
            raise IndexError("Cannot call twice the [] operator on the FourierArray object.")

        self.check_slice(key)

        sliced_array = FourierArray(self.m_indices, self.n_indices, self.cos_coeff, self.sin_coeff)
        sliced_array.pre_sliced = key
        return sliced_array

    @property
    def shape(self) -> tuple:
        key = self.pre_sliced
        if isinstance(self.pre_sliced, type(None)):
            key = slice(None)

        valid_array = self.cos_coeff if self.cos_coeff.any() else self.sin_coeff
        outer_shape = valid_array[..., 0][key].shape
        return outer_shape

    def __call__(
        self,
        theta: Union[List, np.ndarray, Numeric],
        phi: Union[List, np.ndarray, Numeric],
        mode: str = "3d",
        key: slice = None,
        derivative: Union[bool, str] = False,
    ) -> np.ndarray:
        """
        Evaluate the array at given points theta, phi

        Automatically determines the available coefficients (sin, cos or both)

        Arguments
        ---------
            theta: poloidal angles (can be a number or a list of angles)
            phi : toroidal angles (can be a number or a list of angles)
            mode: if "3d", then all the harmonics are used. if "2d", then only n=0 harmonics are used
            key: if set, then the values are calculated only on the slice defined by key
            derivative: if set to "theta", calculates the directional derivate in theta
            if set to "phi", calculates the directional derivative in phi. Other value are ignored

        Return
        ------
            returns a (..., theta_dim, phi_dim) array of the calculated points
            ... indicates the outer dimensions of the input coefficients
            The array is flattened where len(dim) == 1
        """

        if isinstance(key, type(None)):
            key = self.pre_sliced
        if isinstance(key, type(None)):
            key = slice(None)

        if not check_if_iterable(theta):
            theta = np.array([theta])
        if not check_if_iterable(phi):
            phi = np.array([phi])

        if theta.ndim > 1 or phi.ndim > 1:
            raise ValueError("Angles cannot be more than 1-dimensional")

        # Check if slice is valid with dimensions of coefficients
        self.check_slice(key)

        # Find the dimensions of output array
        valid_array = self.cos_coeff if self.cos_coeff.any() else self.sin_coeff
        outer_shape = valid_array[..., 0][key].shape
        if outer_shape == ():
            outer_shape = (1,)

        # It is more probable to have theta values than phi
        # The last two dimensions will be inverted at the end
        # of the calculation
        out_shape = (*outer_shape, phi.shape[0], theta.shape[0])
        array_out = np.zeros(out_shape).astype(np.float64)

        # Short-hand for lisibility
        m = self.m_indices.astype(int)
        n = self.n_indices.astype(int)
        # To keep the dimensionnality of the arrays at all times
        # We need to replace singular ints into slice
        def to_slice(num: int) -> slice:
            last = num + 1
            if num == -1:
                last = None
            return slice(num, None)

        corrected_key = key
        if isinstance(key, tuple):
            corrected_key = []
            for dim in key:
                if isinstance(dim, int):
                    corrected_key.append(to_slice(dim))
                else:
                    corrected_key.append(dim)
            corrected_key = tuple(corrected_key)
        elif isinstance(key, int):
            corrected_key = to_slice(key)
        else:
            corrected_key = key
        if self.cos_coeff.any():
            C = self.cos_coeff[corrected_key]
        else:
            C = np.array([]).astype(np.float64)
        if self.sin_coeff.any():
            S = self.sin_coeff[corrected_key]
        else:
            S = np.array([]).astype(np.float64)

        if C.any() and S.any():
            fast_sincos(
                array_out,
                outer_shape,
                theta.astype(np.float64),
                phi.astype(np.float64),
                n.astype(int),
                m.astype(int),
                C.astype(np.float64),
                S.astype(np.float64),
                mode,
            )
        elif C.any():
            fast_cos(
                array_out,
                outer_shape,
                theta.astype(np.float64),
                phi.astype(np.float64),
                n.astype(int),
                m.astype(int),
                C.astype(np.float64),
                mode,
            )
        else:
            fast_sin(
                array_out,
                outer_shape,
                theta.astype(np.float64),
                phi.astype(np.float64),
                n.astype(int),
                m.astype(int),
                S.astype(np.float64),
                mode,
            )
        return np.moveaxis(array_out, -1, -2).squeeze()

        if derivative == "theta":
            pass
        elif derivative == "phi":
            pas
