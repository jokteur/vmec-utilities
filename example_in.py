from VMEC_Reader.input import InputFile, InputVariable, IndexedArray

# The input file is made of two sections (indata, bootin) with the following groups:
# INDATA
#   control, grid, freeboundaries, pressure, flow, current, boundary
# BOOTIN
#   misc (contains all variables)
input = InputFile("input.kink.in")

# The names here are case insensitive

# Print a section
print(input.indata)

# Print a group
print(input.INDATA.boundary)  # or print(input.boundary)

# Print a variable
print(input.indata.boundary.rbc)  # or print(input.boundary.rbc) or print(input.rbc)

# Remove a variable from input file
input.remove("precon_type")  # alternatively input.indata.remove("precon_type")

# or a section
input.remove("bootin")

# Remove multiple variables
input.remove(["ntheta", "mpol"])
# can also use wildcards (only for variables):
input.remove("n*")

# Add a variable
input.indata.Control.set_variable("test", data=[1, 2, 3], type=int)

# Modify a variable.
# A variable is an InputVariable class, which contains
# the name, the type, , the description of the variable
input.delt.data = 0.6

# Special variables called IndexedArray are used when we want to specify an array with indices
# See documentation of IndexedArray for more infos

# Here we are removing all indices (x,y) with
# x < 0
input.rbc.data.remove(("<0", "*"))
# Can have multiple removal rules
input.rbc.data.remove([(">4", "*"), ("*", "<0"), ("*", ">4")])
print(input.rbc.data.indices)

# Save the new modified input file
input.to_file("input.modified.in", with_comments=True)
