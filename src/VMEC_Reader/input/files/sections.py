sections = {
    "control": ["precon_type", "prec2d_threshold", "delt", "nstep", "niter_array", "ns_array", "ftol_array"],
    "grid": ["lasym", "lrfp", "nfp", "mpol", "ntor", "ntheta", "nzeta"],
    "free boundary": ["lfreeb", "mgrid_file", "extcur"],
    "pressure": ["gamma", "pres_scale", "pmass_type", "am", "am_aux_s", "am_aux_f"],
    "flow": [],
    "current": ["ncurr", "piota_type", "ai", "ai_aux_s", "ai_aux_f"],
    "boundary": ["phiedge", "raxis", "zaxis", "rbc", "zbs", "rbs", "zbc"]
}

def varname_to_section(varname: dict) -> str:
    for section, variables in sections.items():
        if varname in variables:
            return section
    return "other"