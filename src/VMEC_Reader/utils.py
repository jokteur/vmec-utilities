from typing import Any


def check_if_iterable(element: Any) -> bool:
    try:
        _ = iter(element)
    except TypeError:
        return False
    else:
        return True
