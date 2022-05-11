# VMEC input and output reader
A tool for manipulating VMEC input files and output files.

## Installation
It is recommended to create a virtual environmnent to run the code of this repository. This will avoid package conflicts with other python projects.

```
python -m venv path_for_the_venv
source path_for_the_venv
pip install --upgrade wheel
```

or if you use anaconda, you can create the environnement as follows:

```
conda create --name myenv python=3.9
```

Once the environnement is created, activate the environment (through `source path_for_the_venv` or through `conda activate myenv`), clone the repository and install:
```
cd VMEC_Reader
pip install .
```

or if you want to developp the library in this repository:
```
pip install -e .
```
.