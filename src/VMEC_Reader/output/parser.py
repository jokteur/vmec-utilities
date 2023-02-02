from dataclasses import dataclass

import numpy as np
from typing import Any, Dict, Type
from scipy.io import netcdf_file


from .description import try_get_description

class WoutVariable(np.ndarray):
    def __new__(cls, input_array, description=None):
        obj = np.asarray(input_array).view(cls)
        obj.description = description
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.description = getattr(obj, 'description', None)

    def __repr__(self) -> str:
        data_repr = super().__repr__()
        return f"WoutVariable(\n  data: {data_repr},\n  description: {self.description}\n)"
    
    def __str__(self) -> str:
        data_repr = super().__str__()
        return f"WoutVariable(\n  data: {data_repr},\n  description: {self.description}\n)"


class DerivedVariables:
    def __init__(self, wout: "WoutFile") -> None:
        self.wout = wout

    def getRadialVariable(self, sqrt=True):
        """
        Returns the correctly calculated radial variable
        """
        S = None
        # Poloidal
        if self.wout.lrfp__logical__.data:
            S = self.wout.chi.data / self.wout.chi.data[-1]
        # Toroidal
        else:
            S = self.wout.phi.data / self.wout.phi.data[-1]

        if sqrt:
            S = np.sqrt(S)
        return S


class WoutFile:
    """
    Class for loading a netCDF VMEC Wout file, along with the descriptions of all
    the out variables
    """

    def __init__(self, file: str) -> None:
        """
        Arguments
        ---------
            file: netCDF VMEC Wout file
        """
        self.__file = file
        self.__vars = {}

        wout = netcdf_file(file)
        self.__names = list(wout.variables.keys())

        self.derived: DerivedVariables = DerivedVariables(self)
        wout.close()

    def getDescriptions(self) -> Dict[str, str]:
        """Returns all the variables with the names and descriptions."""
        return {v: try_get_description(v) for v in self.__names}

    def __getattr__(self, var_name: str) -> Type["WoutVariable"]:
        # In case the user still wants to access the class internal variables
        if var_name in ["__file", "__vars", "__names"]:
            return vars(self)[var_name]

        # Variable has already been called before
        if var_name in self.__vars:
            return self.__vars[var_name]
        else:
            if var_name not in self.__names:
                raise AttributeError(f"WoutFile does not have attribute '{var_name}'")

            wout = netcdf_file(self.__file)
            self.__vars[var_name] = WoutVariable(
                wout.variables[var_name][()].copy(), try_get_description(var_name)
            )
            wout.close()
            return self.__vars[var_name]

    def __repr__(self) -> str:
        return f"WoutFile ('{self.__file}')"

    def __str__(self) -> str:
        out = f"WoutFile ('{self.__file}')\n"
        for name, description in self.getDescriptions().items():
            out += f"  {name}: {description}\n"
        return out[:-1]


def binaryToStr(array: np.array):
    """Converts a numpy binary array to string"""
    out = ""
    for a in array:
        out += a.decode("utf-8")
    return out
