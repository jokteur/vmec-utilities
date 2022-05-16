"""
Author: Joachim Koerfer, 2022

File for loading the descriptions of the wout variables
"""
from numpy import short
import toml
from typing import Dict, Any
import os


path = os.path.dirname(__file__)
# In the fortran file, we have var_name => netcdf_name
short_names = toml.load(os.path.join(path, "files", "out_short_names.toml"))
# In the long names, we have var_name => description
long_names = toml.load(os.path.join(path, "files", "out_long_names.toml"))


def read_names():
    """
    Reads the output variable names and returns a dictionary with
    the associated descriptions.
    """

    descriptions = {}
    for k, v in short_names.items():
        if k in long_names:
            descriptions[v] = long_names[k]
        else:
            descriptions[v] = k
    return descriptions


# Using a function keeps the namespace clear
descriptions = read_names()


def try_get_description(var_name: str) -> str:
    """
    In the case a description is missing, this function tries to match
    to a variable anyway.
    """

    if var_name in descriptions:
        return descriptions[var_name]
    elif "__logical__" in var_name:
        short_name = var_name[:-11]
        if short_name in descriptions:
            return descriptions[short_name]

    return "Did not find any description"
