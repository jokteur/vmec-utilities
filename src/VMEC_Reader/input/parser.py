"""
Author: Joachim Koerfer, 2022
"""
from copy import deepcopy
from collections import OrderedDict
import re
from typing import Any, Dict, List, Tuple, Union
import numpy as np

from .data import IndexedArray


def float_or_int(input: str) -> Union[float, int]:
    """
    Tries to detect if a string is an int or a float and returns it
    """
    num = None
    try:
        num = int(input)
    except ValueError:
        try:
            num = float(input)
        except ValueError:
            return input
    return num


def strdata_to_data(input: str) -> Any:
    if "'" in input:
        return input.strip("'")
    elif input in ["T", ".true."]:
        return True
    elif input in ["F", ".false."]:
        return False
    elif len(input.split()) > 1:
        return np.array([float_or_int(num) for num in input.split()])
    else:
        return float_or_int(input)


def collect_input_variables(input_path: str) -> Dict[str, Dict[str, str]]:
    """
    Reads a VMEC input file and returns a dict with all the variables and their data

    Note
    ----
    What counts as a valid input file is not trivial.

    ```
    ns_array = 25 73 ! arrays can be separated by a new line
    223 227
    niter_array = 2999 3999 3999 5999,
    some_war = 'tricky text with \'= ,', other_war = 1e-3
    ```

    is an example of a tricky input file, which should give a dict like:

    ```
    ns_array = [25 73 223 227]
    niter_array = [2999 3999 3999 5999]
    some_war = 'tricky test with \'= ,'
    other_war = 1e-3
    ```
    """
    variables: Dict[str, Dict[str, str]] = OrderedDict()

    section_name = ""

    file = open(input_path, "r")
    lines_with_comments = file.readlines()

    # Remove comments, removes complexity further down
    lines: List[str] = []
    for line in lines_with_comments:
        # Strip comments
        line = line.split("!")[0]
        if not line:  # Means that the whole line is a comment
            continue
        lines.append(line)

    # Remove empty lines and spaces
    lines = [line.strip() for line in lines if line.strip()]
    file_string = "\n".join(lines)

    class Accumulator:
        def __init__(self) -> None:
            self.accumulator = ""
            self.counter = 0
            self.counter_activated = False
            self.skip = False

        def add(self, chr: str):
            """
            Adds character to the accumulator except if
            Accumulator.get() was called beforehand.

            This function also replaces "\n" by " "
            """
            if chr == "\n":
                chr = " "
            if not self.skip:
                self.accumulator += chr
            else:
                self.skip = False
            if self.counter_activated:
                self.counter -= 1

        def get(self, return_tuple=False) -> Union[str, Tuple[str, str]]:
            """
            Returns the value of the accumulator
            and resets it.

            Arguments
            ---------
                no_tuple_return: if set to True, will return a string in any case
                    even if the counter is activated
            """
            self.skip = True
            if self.counter_activated and return_tuple:
                before = self.accumulator[: self.counter]
                after = self.accumulator[self.counter :]
                self.counter = 0
                self.counter_activated = False
                self.accumulator = ""
                return before, after
            else:
                tmp = self.accumulator
                self.accumulator = ""
                self.counter = 0
                self.counter_activated = False
                return tmp

        def reset(self):
            """Resets the accumulator."""
            self.counter = 0
            self.counter_activated = False
            self.accumulator = ""
            self.skip = True

        def is_flag_set(self):
            return self.counter_activated

        def set_flag(self):
            """
            Sets a flag at this place and activates the counter.

            Useful for going back.
            """
            self.counter_activated = True
            self.counter = 0

    flag = "normal"  # section_name, normal, var_name, data, string, skip
    section_name = ""
    var_name = ""
    accumulator = Accumulator()

    # We have to go character by character to parse the file
    for chr in file_string:
        new_line = chr == "\n"

        # Sections: begin with &SECTION_NAME and ends with /
        if flag == "normal":
            if chr == "&":  # begin section
                accumulator.reset()
                flag = "section_name"
            if chr == "/":  # end section
                flag = "normal"
                section_name = ""
        elif flag == "section_name":
            if new_line:
                flag = "var_name"
                section_name = accumulator.get()
                variables[section_name] = OrderedDict()

        # Varname can only stop when we encounter a =
        elif flag == "var_name":
            if chr == "=":
                flag = "data"
                var_name = accumulator.get()
        elif flag == "data":
            if chr == "'":  # Beginning of a new string
                flag = "string"
            elif chr == ",":
                if not accumulator.is_flag_set():
                    flag = "var_name"
                    variables[section_name][var_name] = accumulator.get()
            elif chr == "=":
                if not accumulator.is_flag_set():
                    raise ValueError(f"Parsing error at '{accumulator.accumulator}'")
                data, next_var_name = accumulator.get(return_tuple=True)
                variables[section_name][var_name] = data
                var_name = next_var_name
            elif new_line:
                # We may or may not encounter a '=' on the next line
                # If this happens, then we were in 'var_name' mode not in data
                # mode after the new line
                accumulator.set_flag()
            # If we arrive at the end of a section but no ',' has been used
            elif accumulator.is_flag_set() and chr == "/":
                variables[section_name][var_name] = accumulator.get()
                flag = "normal"
        # In string mode, only a non-escaped ' can finish the mode
        elif flag == "string":
            if chr == "\\":  # Escaping characters means that we skip the next character
                flag = "skip"
            elif chr == "'":
                flag = "data"
        elif flag == "skip":
            flag = "string"

        accumulator.add(chr)

    typed_variables: Dict[
        str, Dict[str, Union[float, int, str, IndexedArray, np.np.ndarray]]
    ] = OrderedDict()

    # Once we have parsed the file, we can assign and type the different variables
    for section, dic in variables.items():
        typed_variables[section] = OrderedDict()
        for var_name, data in dic.items():
            data = data.strip().strip(",")
            var_name = var_name.strip()

            # Special case of var(i,j,k,...) variables
            special_var = re.split("(\([-0-9, ]+\))", var_name)
            if len(special_var) == 3:
                var_name = special_var[0]
                index_tuple = tuple([int(val) for val in special_var[1][1:-1].split(",")])
                if var_name not in typed_variables[section]:
                    typed_variables[section][var_name] = IndexedArray(len(index_tuple))
                # Does not support comment right now
                typed_variables[section][var_name].add(strdata_to_data(data), index_tuple)
            else:
                # Test the different types of variables
                typed_variables[section][var_name] = strdata_to_data(data)

    return typed_variables
