from dataclasses import dataclass
from pydoc import writedoc
from typing import Any, Dict, List, Tuple, Type, Union
import numpy as np

from ..utils import check_if_iterable


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


@dataclass
class InputVariable:
    """
    Stores an input variable, along with its description, type
    and size.
    """

    data: Union[np.ndarray, Type["IndexedArray"], int, float, str]
    type: type
    size: str
    description: str


class TupleComparator:
    def __init__(self, tuple_) -> None:
        self.tuple_ = tuple_

    def __lt__(self, other) -> bool:
        for i, j in zip(self.tuple_, other.tuple_):
            if i < j:
                return True
            elif i > j:
                return False
        return False


class IndexedArray:
    """
    Stores an array with an index array (which can have negative values)
    """

    dimension: int
    array: List[float] = []
    indices: List[Tuple] = []
    descriptions: List[str] = []

    def __init__(
        self,
        dimension,
        values: Union[List, np.ndarray, float, int] = [],
        indices: Union[Tuple, List[Tuple]] = [],
        descriptions: Union[str, List[str]] = "",
    ) -> None:
        self.dimension = dimension
        self.array = []
        self.indices = []
        self.idx_tuples = dict()
        self.descriptions = []
        self.add(values, indices, descriptions)

    def add(
        self,
        values: Union[List, np.ndarray, float, int],
        indices: Union[Tuple, List[Tuple]],
        descriptions: Union[str, List[str]] = "",
    ):
        """
        Adds values to the indexed array.

        Replaces existing values at indices if specified index already registered

        Arguments
        ---------
            values: number of array_like of the values to be added to the array
            indices: tuple of indices or list of tuple of indices. If a List is provided in values,
            then a list of indices with the same length as values should be provided
            descriptions: if a str is provided, then the same description is given to all indices
        """
        is_tuple = isinstance(indices, tuple)
        is_iter = check_if_iterable(values)
        if is_iter and is_tuple or not is_iter and not is_tuple:
            raise ValueError("Both arguments values and indices must be iterable.")
        if not is_iter:
            values = [values]
            indices = [indices]

        if isinstance(descriptions, str):
            descriptions = [descriptions] * len(values)

        if len(values) != len(indices):
            raise ValueError("Value and indices should have the same length")
        if len(values) != len(descriptions):
            raise ValueError("Value and descriptions should have the same length")

        for val, idx, desc in zip(values, indices, descriptions):
            if len(idx) != self.dimension:
                raise ValueError(
                    f"Dimensionality of indices does not match the IndexedArray({self.dimension})"
                )
            if str(idx) in self.idx_tuples:
                array_idx = self.idx_tuples[idx]
                self.array[array_idx] = val
                self.descriptions[array_idx] = desc
            else:
                self.array.append(val)
                self.indices.append(idx)
                self.descriptions.append(desc)
                self.idx_tuples[str(idx)] = len(self.array)

    def __repr__(self) -> str:
        out = "IndexedArray("
        short_hand = len(self.array) > 10
        write_dots = True
        for i, (val, idx, comment) in enumerate(zip(self.array, self.indices, self.descriptions)):
            if short_hand:
                if i > 4 and i < len(self.array) - 5:
                    if write_dots:
                        out += " [...] "
                        write_dots = False
                    continue

            if i != 0:
                out += ", "
            out += f"{idx}: {val}"
            if comment:
                out += f" ! {comment}"
        out += ")"
        return out

    def match_rule(self, rule: Tuple, index: Tuple) -> bool:
        """
        Returns true if the specified rule matches the index tuple.
        """
        num_matches = 0
        num_rules = 0
        for r, i in zip(rule, index):
            if isinstance(r, str):
                if "*" in r:
                    num_matches += 1
                    num_rules += 1
                else:
                    split = r.split("&")
                    for s in split:
                        num_rules += 1
                        try:
                            if ">=" in s:
                                num = int(s.split(">=")[1])
                                num_matches += int(i >= num)
                            elif "<=" in s:
                                num = int(s.split("<=")[1])
                                num_matches += int(i <= num)
                            elif ">" in s:
                                num = int(s.split(">")[1])
                                num_matches += int(i > num)
                            elif "<" in s:
                                num = int(s.split("<")[1])
                                num_matches += int(i < num)
                            else:
                                num_matches += int(int(s) == i)
                        except TypeError:
                            pass
            else:
                num_matches += int(r == i)

        if num_matches == num_rules:
            return True
        return False

    def remove(self, rules: Union[Tuple[Union[int, str]], List[Tuple[Union[int, str]]]]):
        """
        Remove some indices from the array with a set of rules

        Arguments
        ---------
            rules: a tuple of str (or a list of tuple) of str which indicates for each dimension
            the rule(s) for removal

        Rules for removal
        -----------------

        Specify simple numbers to remove, such as: (2, 3) will remove the couple (2, 3) index in
        the array

        It is possible to use wildcard, such as: (2, "*"), which will remove all indices with 2 in
        the first dimension

        It is possible to have a complex rule, such as: (">-1&<=3", "*"), which will remove all
        indices greater than -1 but less or equal than 3 in the first dimension

        The sign for number comparison should always be left to the indicated number.
        E.g. ">=-1" is valid, but "1>=" is not.

        Notes
        -----
        Executes in O(n*k*d), n is the size of the IndexedArray, k the number of rules and d the dimensionality
        As generally n >> k or d, we have O(n)
        """
        if isinstance(rules, tuple):
            rules = [rules]

        new_array = []
        new_indices = []
        new_descriptions = []
        new_dic = {}
        for i, idx in enumerate(self.indices):
            num_matches = 0
            for rule in rules:
                if len(rule) != self.dimension:
                    raise ValueError(
                        f"Dimension of rules does not correspond to IndexedArray({self.dimension})"
                    )
                num_matches += int(self.match_rule(rule, idx))
            if num_matches == 0:
                new_array.append(self.array[i])
                new_indices.append(idx)
                new_descriptions.append(self.descriptions[i])
                new_dic[str(idx)] = len(new_array)

        self.array = new_array
        self.indices = new_indices
        self.descriptions = new_descriptions
        self.idx_tuples = new_dic

    def get_ordered(self) -> Tuple:
        """Returns the indices, array and descriptions ordered by
        the indices"""

        new_indices = argsort([TupleComparator(idx) for idx in self.indices])
        ordered_array = [self.array[idx] for idx in new_indices]
        ordered_indices = [self.indices[idx] for idx in new_indices]
        ordered_descriptions = [self.descriptions[idx] for idx in new_indices]

        return ordered_indices, ordered_array, ordered_descriptions
