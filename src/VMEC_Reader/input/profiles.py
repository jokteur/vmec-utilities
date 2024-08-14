from dataclasses import dataclass
import numpy as np

from .parameters import InputFile, InputVariable


@dataclass
class Profile:
    # pressure, mass, iota, safety, rotation, temperature
    name: str
    # unit of the profile
    unit: str
    # power_series, cubic_spline, todo
    ptype: str
    # rho_sq, rho. VMEC uses rho_sq = Psi / Psi_b representation, but we can convert it to rho = sqrt(Psi / Psi_b)
    radial_coordinate: str
    # in case of power_series, the x values are not used
    xs: np.ndarray
    # beware that in case of power series, in vmec the coefficients are in reverse order. Here we use numpy's convention
    data: np.ndarray

    def to_rho_representation(self, poly_resolution: int = None):
        """Converts the profile to a rho= sqrt(Psi / Psi_b) representation"""
        out = Profile(self.name, self.unit, self.ptype, "rho", self.xs, self.data)

        if self.radial_coordinate == "rho":
            return out

        if self.ptype == "power_series":
            xs = np.linspace(0, 1, 100)
            ys = np.polyval(self.data, xs)
            if not poly_resolution:
                poly_resolution = len(self.data)
            out.data = np.polyfit(np.sqrt(xs), ys, poly_resolution)
        elif self.ptype == "cubic_spline":
            out.xs = np.sqrt(self.xs)

        return out

    def to_rho_sq_representation(self):
        """Converts the profile to a rho_sq = Psi / Psi_b representation"""
        out = Profile(self.name, self.unit, self.ptype, "rho_sq", self.xs, self.data)

        if self.radial_coordinate == "rho_sq":
            return out

        if self.ptype == "power_series":
            xs = np.linspace(0, 1, 100)
            ys = np.polyval(self.data, xs)
            out.data = np.polyfit(xs**2, ys, len(self.data))
        elif self.ptype == "cubic_spline":
            out.xs = self.xs**2

        return out

    def to_ptype(self, new_ptype: str, x_resolution: int = 99, poly_resolution: int = 10):
        """Change underlying representation of the profile (power_series, cubic_spline)"""
        out = Profile(self.name, self.unit, new_ptype, self.radial_coordinate, self.xs, self.data)

        if self.ptype == new_ptype:
            return out

        if new_ptype == "power_series":
            out.data = np.polyfit(self.xs, self.data, poly_resolution)
            out.xs = None
        elif new_ptype == "cubic_spline":
            out.xs = np.linspace(0, 1, x_resolution)
            out.data = np.polyval(self.data, out.xs)

        return out

    def transform(self, func, poly_resolution: int = None, new_unit: str = ""):
        """Transforms the profile using a custom function

        Arguments:
          func: function to apply to the profile
          poly_resolution: resolution of the polynomial fit in case of power_series
          new_unit: new unit of the profile (if None, keep the current unit)
        """
        out = Profile(self.name, new_unit, self.ptype, self.radial_coordinate, self.xs, self.data)
        if self.ptype == "cubic_spline":
            out.data = func(out.data)
        elif self.ptype == "power_series":
            if not poly_resolution:
                poly_resolution = len(self.data)
            ys = func(np.polyval(self.data, self.xs))
            out.data = np.polyfit(self.xs, ys, poly_resolution)

        return out
    
    def get_xy(self, x_resolution: int = 100):
        """Returns the x and y values of the profile"""
        if self.ptype == "power_series":
            xs = np.linspace(0, 1, x_resolution)
            ys = np.polyval(self.data, xs)
        elif self.ptype == "cubic_spline":
            xs = self.xs
            ys = self.data

        return xs, ys


def _get_mass_or_pressure(input: InputFile, name: str) -> Profile:
    try:
        input.pmass_type
    except AttributeError:
        raise AttributeError("The input file should have a pmass_type variable")

    xs = None
    data = None
    if input.pmass_type.data == "power_series":
        data = input.am.data[::-1]
    elif input.pmass_type.data == "cubic_spline":
        xs = input.am_aux_s.data
        data = input.am_aux_f.data

    unit = "Pa"
    if name == "mass":
        unit = "kg/m^3"

    return Profile(name, unit, input.pmass_type.data, "rho_sq", xs, data)


def _get_iota_or_safety(input: InputFile, name: str) -> Profile:
    try:
        input.piota_type
    except AttributeError:
        raise AttributeError("The input file should have a piota_type variable")

    xs = None
    data = None

    if input.piota_type.data == "power_series":
        data = input.ai.data[::-1]
    elif input.piota_type.data == "cubic_spline":
        xs = input.ai_aux_s.data
        data = input.ai_aux_f.data

    try:
        input.lrfp
    except AttributeError:
        raise AttributeError("The input file should have a lrfp variable to specify safety or iota")
    is_iota = True

    # If LRFP=T then q=1/iota profile is specified
    if input.lrfp.data:
        is_iota = False

    return Profile(name, "", input.piota_type.data, "rho_sq", xs, data), is_iota


def get_pressure(input: InputFile) -> Profile:
    """Returns the pressure profile of in the input file (in Pascal)"""
    if input.gamma.data != 0:
        raise ValueError(
            "As gamma is not 0, the mass profile has been defined instead of the pressure profile. Please use the get_mass function instead"
        )

    return _get_mass_or_pressure(input, "pressure")


def get_mass(input: InputFile) -> Profile:
    if input.gamma.data == 0:
        raise ValueError(
            "As gamma is 0, the pressure profile has been defined instead of the mass profile. Please use the get_pressure function instead"
        )

    return _get_mass_or_pressure(input, "mass")


def get_safety(input: InputFile) -> Profile:
    out, is_iota = _get_iota_or_safety(input, "safety")

    if is_iota:
        return out.transform(lambda x: 1 / x)
    else:
        return out


def get_iota(input: InputFile) -> Profile:
    out, is_iota = _get_iota_or_safety(input, "iota")

    if is_iota:
        return out
    else:
        return out.transform(lambda x: 1 / x)


def get_rotation(input: InputFile) -> Profile:
    print("Warning: Feature not implemented yet")


def get_temperature(input: InputFile) -> Profile:
    print("Warning: Feature not implemented yet")
