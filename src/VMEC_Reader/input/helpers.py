from .parameters import InputFile, InputVariable
from .profiles import Profile, set_profile

def set_poloidal_flux_mode(input_file: InputFile, safety: Profile, pressure: Profile):
    """
    Sets the radial variable as the poloidal flux in VMEC. 
    
    The poloidal flux is defined as:
    $$

    $$
    """
    input_file.set_variable("lrfp", True, bool)

    set_profile(input_file, safety, "piota", "ai")
    set_profile(input_file, pressure, "pmass", "am")

def set_toroidal_flux_mode(input_file: InputFile, iota: Profile, pressure: Profile):
    """
    Sets the radial variable as the toroidal flux in VMEC.
    """
    input_file.set_variable("lrfp", False, bool)

    set_profile(input_file, iota, "piota", "ai")
    set_profile(input_file, pressure, "pmass", "am")