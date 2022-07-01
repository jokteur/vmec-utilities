"""
Read coil file for running xgrid
"""

from collections import defaultdict
from typing import Dict, List, Union
import numpy as np
from dataclasses import dataclass


@dataclass
class Coil:
    """
    Contains a series of points describing a coil
    """

    points: np.ndarray
    currents: float


@dataclass
class CoilGroup:
    name: str
    coils: List[Coil]

    def __repr__(self) -> str:
        return f"CoilGroup(name={self.name}, num_coils={len(self.coils)})"


class CoilFile:
    """
    Class for storing and representing a coil file
    """

    def __init__(self, file: str = "") -> None:
        """
        Reads a coil file and stores in a class.

        Assumes a file in the form:
        ```
        periods N
        begin filament
        mirror NIL
         [... data]
        end
        ```

        Arguments
        ---------
            file: str
                path to the coil file
        """

        self.groups: Dict[int, CoilGroup] = {}
        self.periods = None

        if not file:
            return

        with open(file, "r") as f:
            periods = f.readline().split(" ")[1]
            f.readline()  # Begin filament
            f.readline()  # mirror NIL

            points = []
            currents = []
            line_number = 3
            for line in f.readlines():
                # Remove trailing spaces
                line = " ".join(line.split())
                line_number += 1

                if "end" in line:
                    break

                split = line.split(" ")
                if len(split) != 4 and len(split) != 6:
                    raise ValueError(f"Problem reading numbers at line {line_number}")

                points.append([float(num) for num in split[:3]])
                currents.append(float(split[3]))

                # Finish the description of a coil
                if len(split) == 6:
                    group_number = int(split[4])
                    group_name = split[5]
                    coil = Coil(np.array(points), currents[0])

                    if group_number not in self.groups:
                        self.groups[group_number] = CoilGroup(group_name, [])
                    self.groups[group_number].coils.append(coil)
                    points = []
                    currents = []

    def __repr__(self) -> str:
        return f"CoilFile(groups={list(self.groups.values())})"

    def to_MGRID_file(self, file: str):
        """
        Saves the CoilFile to a compatible MGRID file

        Arguments
        ---------
        file: str
            name of the file to be created
        """

        f = open(file, "w")

        f.writelines(["periods 1\n", "begin filament\n", "mirror NUL\n"])

        for i, (_, group) in enumerate(self.groups.items()):
            for coil in group.coils:
                current = coil.currents
                if isinstance(current, np.ndarray) or isinstance(current, list):
                    current = current[0]
                first_pt = coil.points[0]
                for pt in coil.points:
                    f.write(f"   {pt[0]:.10f} {pt[1]:.10f} {pt[2]:.10f} {current}\n")
                f.write(
                    f"   {first_pt[0]:.10f} {first_pt[1]:.10f} {first_pt[2]:.10f} {0.0:.10f} {i+1} {group.name}\n"
                )
        f.close()
