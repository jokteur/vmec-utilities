import numpy as np
from typing import Any, Union

Numeric: type = Union[int, float, complex, np.number]


def compare_shapes(shape1: tuple, shape2: tuple) -> bool:
    if len(shape1) != len(shape2):
        return False

    for i, j in zip(shape1, shape2):
        if i != j:
            return False

    return True


def check_if_iterable(element: Any) -> bool:
    try:
        _ = iter(element)
    except TypeError:
        return False
    else:
        return True
