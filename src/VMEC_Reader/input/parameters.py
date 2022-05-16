"""
Author: Joachim Koerfer, 2022
"""
import copy
from distutils.errors import CompileError
import os
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Tuple, Type, Union
import numpy as np

from .parser import collect_input_variables
from .data import IndexedArray, InputVariable
from ..utils import check_if_iterable


def str_to_type(type_str: str) -> type:
    if type_str in "integer":
        return int
    if type_str in ["float", "double", "real"]:
        return float
    if type_str in ["logical", "bool"]:
        return bool
    return str


def type_to_str(type_: type, data: Any):
    if type_ is np.ndarray:
        return f"{data}"
    if type_ is int:
        return f"{data}"
    elif type_ is float:
        return f"{data:.12E}"
    elif type_ is bool:
        return "T" if data else "F"
    else:
        return f"'{data}'"


def match_wildcard(pattern: List[str], match_to: str) -> bool:
    """
    Matches a string to another string with wildcard rules

    Arguments
    ---------
        pattern: already splitted pattern. E.g: "ac_*s" would result in the split ["ac_", "s"]
        match_to: string to match to pattern

    Return
    ------
        if the match is successful or not
    """
    if isinstance(pattern, str):
        pattern = pattern.split("*")
    # Match wildcards (*)
    num_matches = 0
    next_idx = 0
    for part in pattern:
        idx = match_to.find(part, next_idx)
        if idx >= 0:
            num_matches += 1
            next_idx = idx + len(part)
    if num_matches == len(pattern):
        return True
    return False


"""
Default variable groups in input parameters

See InputSection() how to set this variable
"""
default_groups = {
    "indata": {
        "control": (
            ["prec*", "delt", "nstep", "*_array"],
            "Control parameters",
        ),
        "grid": (["lasym", "lrfp", "nfp", "mpol", "ntor", "ntheta", "nzeta"], "Grid parameters"),
        "freeboundaries": (["lfreeb", "mgrid_file", "extcur*"], "Free boundaries parameters"),
        "pressure": (
            ["gamma", "pres_scale", "pmass_type", "am*"],
            "Pressure parameters",
        ),
        "flow": (["at", "ah", "bcrit"], "Flow parameters"),
        "current": (
            [
                "ncurr",
                "ai*",
                "ac*",
                "curtor",
                "piota_type",
                "pcurr_type",
            ],
            "Current parameters",
        ),
        "boundary": (
            ["phiedge", "raxis", "zaxis", "rbc*", "rbs*", "zbc*", "zbs*"],
            "Boundary conditions",
        ),
    }
}


class InputGroup:
    """
    InputGroup allows to group input variables for better lisibility

    This class is deeply linked with InputSection
    """

    def __init__(
        self,
        cls: Type["InputSection"],
        variables: List[str],
        description: str,
        no_group_check: bool = False,
    ) -> None:
        """
        cls: reference to mother class
        variables: list of variables contained in the group; can be regular expression
        description: description of the group
        no_group_check: (bool) if no_group_check is true, then we are in the group that contains any variable in cls
        """
        self.cls__ = cls
        self.variables__: List[str] = variables
        self.description__: str = description
        self.no_group_check__: bool = no_group_check

    def match(self, var_name: str) -> bool:
        for var in self.variables__:
            split = var.split("*")
            # Match wildcards (*)
            if len(split) > 1:
                if match_wildcard(split, var_name):
                    return True
            elif var == var_name:
                return True
        return False

    def getVariable(self, var_name: str) -> Type["InputVariable"]:
        var_name = var_name.lower()
        available_vars = self.cls__.variables
        if self.match(var_name) or self.no_group_check__:
            if var_name in available_vars.keys():
                return available_vars[var_name]
            else:
                raise AttributeError(f"Attribute '{var_name}' does not exist")
        else:
            raise AttributeError(f"Attribute '{var_name}' does not exist in '{self.description__}'")

    def addVariable(self, var_name: str, data: Any, type: type):
        var_name = var_name.lower()
        if var_name in self.cls__.variables:
            self.cls__.variables[var_name].data = data
            self.cls__.variables[var_name].type = type
        else:
            self.variables__.append(var_name)
            self.cls__.variables[var_name] = InputVariable(data, type, "", "")

    def __getattr__(self, var_name: str) -> Type["InputVariable"]:
        return self.getVariable(var_name)

    def __repr__(self) -> str:
        return f"InputGroup(Matches: {self.variables__})"

    def remove_match(self, name: str):
        """
        Removes a variable from the group

        Arguments
        ---------
            name: string of the match to remove in group
        """
        if name in self.variables__:
            del self.variables__[self.variables__.index(name)]


class InputSection:
    """
    Loads an input file along with the possible parameters descriptions
    """

    def __init__(
        self,
        name: str,
        input_vars: Dict[str, InputVariable] = OrderedDict(),
        groups: Dict[str, Tuple[List[str], str]] = OrderedDict(),
    ) -> None:
        """
        Initialize a InputSection class.

        Arguments
        ---------
            name: name of the section
            input: file to preload values. If empty, then no default values are assigned
            groups: description of variable groups. Should be a dict with this structure: group_name : (list of variable names, description of group)
        """
        self.variables: Dict[str, InputVariable] = OrderedDict()

        # Describe different groups
        self.groups: Dict[str, InputGroup] = OrderedDict()
        self.name: str = name
        for group_name, (var_list, description) in groups.items():
            self.groups[group_name] = InputGroup(self, var_list, description)
        self.groups["misc"] = InputGroup(self, [], "misc", True)

        # Collect all the descriptions of the input variables
        try:
            path = os.path.dirname(__file__)
            descriptions = open(
                os.path.join(path, "files", f"{name.lower()}_descriptions.txt"), "r"
            )

            for line in descriptions.readlines():
                var_name, var_type, var_size, var_description = line.split(";;;")
                self.variables[var_name.lower()] = InputVariable(
                    None, str_to_type(var_type), var_size, var_description
                )
        except:
            pass

        # Read the input variables
        for var_name, tup in input_vars.items():
            var_name = var_name.lower()
            data, comment = tup[0], tup[1]
            if var_name in self.variables:
                self.variables[var_name].data = data
            else:
                self.variables[var_name] = InputVariable(data, type(data), "", "")
            if comment:
                desc = self.variables[var_name].description
                if desc:
                    self.variables[var_name].description = desc.strip() + " ; " + comment
                else:
                    self.variables[var_name].description = comment

    def __getattr__(self, var_name: str) -> Union[InputGroup, InputVariable]:
        """
        Searches in the groups and then the input variables to find a suitable attribute

        Note
        ----
        It is case insensitive
        """
        var_name = var_name.lower()
        if var_name in self.groups:
            return self.groups[var_name]
        elif var_name in self.variables.keys():
            return self.variables[var_name]
        else:
            raise AttributeError(f"InputSection does not have the attribute '{var_name}'")

    def __repr__(self) -> str:
        group_str = ", ".join(list(self.groups.keys()))
        var_names = list(self.variables.keys())
        variables_str = ""

        print_dots = True
        for i, name in enumerate(self.variables.keys()):
            if len(var_names) > 10:
                if i > 4 and i < len(var_names) - 5:
                    if print_dots:
                        variables_str += " [...] "
                        print_dots = False
                    continue
            if i != 0:
                variables_str += ", "
            variables_str += name

        return f"InputSection(Groups: {group_str} / Variables: {variables_str})"

    def remove(self, var_name: str):
        """
        Searches the variable(s) in the section and removes it (with wildcards *)
        """
        to_remove = []
        var_name = var_name.lower()
        for name in self.variables.keys():
            if match_wildcard(var_name, name):
                to_remove.append(name)

        for name in to_remove:
            del self.variables[name]
            # Remove the full variable names in group
            for group in self.groups.values():
                if name in group.variables__:
                    idx = group.variables__.index(name)
                    del group.variables__[idx]

    def remove_group(self, name: str, keep_variables=False):
        pass

    def to_string(self, with_comments=False) -> str:
        """
        Writes the variables of the class to a string (compatible VMEC input file)
        """
        # Remove all unvalid data (i.e. None)
        variables = copy.deepcopy(self.variables)
        for var_name, var in self.variables.items():
            if isinstance(var.data, type(None)):
                del variables[var_name]

        # Assign groups to valid variables
        group_variables = OrderedDict()
        available_vars = set(variables.keys())

        for group_name, group in self.groups.items():
            iterator = group.variables__
            if group_name == "misc":
                iterator = ["*"]
            for var_name in iterator:
                split = var_name.split("*")
                remove_vars = []
                # As the names in group can be with wild card, we have to check for it first
                if len(split) > 1:
                    # Look for potential matches
                    for potential_match in available_vars:
                        if match_wildcard(split, potential_match):
                            group_variables.setdefault(group_name, []).append(potential_match)
                            remove_vars.append(potential_match)
                elif var_name in variables:
                    group_variables.setdefault(group_name, []).append(var_name)
                    remove_vars.append(var_name)

                for var in remove_vars:
                    if var in available_vars:
                        available_vars.remove(var)

        separator = "-" * 20
        out = f"&{self.name.upper()}\n"  # Begin of section
        for group_name, var_names in group_variables.items():
            group = self.groups[group_name]

            # Write the group descriptions first
            out += f"\n!{group.description__.upper()}\n"
            out += f"!{separator}\n"
            # Then write all the variables
            for var_name in var_names:
                variable = self.variables[var_name]

                # Special case of indexed arrays
                if isinstance(variable.data, IndexedArray):
                    indexed_array: IndexedArray = variable.data
                    for index, value, comment in zip(*indexed_array.get_ordered()):
                        tup_str = ",".join(str(x) for x in index)
                        out += (
                            f"{var_name.upper()}({tup_str}) = {type_to_str(variable.type, value)}"
                        )
                        if comment.strip():
                            out += f", ! {comment.strip()}\n"
                        else:
                            out += ",\n"
                    continue

                out += f"{var_name.upper()} = "
                # variable.data can be a (int, float, string, bool) or a sequence of these things
                # We must first check string, because it is considered iterable by Python
                if isinstance(variable.data, str):
                    out += type_to_str(variable.type, variable.data) + " "
                elif check_if_iterable(variable.data):
                    for el in variable.data:
                        out += type_to_str(variable.type, el) + " "
                else:
                    out += type_to_str(variable.type, variable.data) + " "

                if with_comments and variable.description.strip():
                    out += f", ! {variable.description.strip()} \n"
                else:
                    out += ",\n"
        out += "/"  # End of section

        return out


class InputFile:
    """
    Class for writing an input file, with multiple sections
    """

    def __init__(
        self, path: str, groups: Dict[str, Dict[str, Tuple[List[str], str]]] = default_groups
    ) -> None:
        """Builds an input file with multiple sections from a file.

        Arguments
        ---------
            path: path to the input file
            default_groups:
        """
        variables = collect_input_variables(path)

        self.sections: Dict[str, InputSection] = OrderedDict()

        for section_name, input_vars in variables.items():
            section_name = section_name.lower()
            if section_name in groups:
                section_group = groups[section_name]
            else:
                section_group = OrderedDict()

            self.sections[section_name] = InputSection(section_name, input_vars, section_group)

    def __getattr__(self, attr: str) -> Union[InputSection, InputGroup, InputVariable]:
        """
        Finds first a section with the name, then an input group in all the sections,
        and then a variable name in all the sections.
        """
        # First search in sections
        attr = attr.lower()
        if attr in self.sections:
            return self.sections[attr]
        else:
            # Then in the groups
            for section in self.sections.values():
                if hasattr(section, attr):
                    return section.__getattr__(attr)

            # And finally in the variable names
            for section in self.sections.values():
                if attr in section.variables.keys():
                    return section.variables[attr]

        raise AttributeError(f"Could not find '{attr}' anywhere in the input file.")

    def add_section(self, name: str, input_vars: Dict[str, InputVariable] = OrderedDict()):
        """
        Adds a section to the input file.
        """
        if name not in self.sections:
            self.sections[name] = InputSection(name, input_vars)

    def to_file(self, path: str, with_comments=False):
        file = open(path, "w")
        for section in self.sections.values():
            file.write(section.to_string(with_comments))
            file.write("\n\n")
        file.write("&END")
        file.close()

    def remove_section(self, name):
        to_remove = []
        for key in self.sections.keys():
            if match_wildcard(name, key):
                to_remove.append(key)
        for key in to_remove:
            self.sections.pop(key)

    def remove(self, names: Union[str, Iterable[str]]):
        """Erases a variable, group or section from the class

        If a group is removed, all the variables contained in the group are removed

        Arguments
        ---------
            names: names of the variable, group or section
                  usage of wildcards "*" is authorized
        """
        if isinstance(names, str):
            names = [names]

        for name in names:
            name = name.lower()
            if "*" in name:
                for section in self.sections.values():
                    section.remove(name)
            else:
                attr = self.__getattr__(name)
                if isinstance(attr, InputVariable):
                    for section in self.sections.values():
                        section.remove(name)
                elif isinstance(attr, InputGroup):
                    if name == "misc":
                        raise ValueError("Cannot remove the group 'misc' from the input section.")
                    for section in self.sections.values():
                        section.remove_group(name)
                elif isinstance(attr, InputSection):
                    self.remove_section(name)
