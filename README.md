![grates logo](https://github.com/akvas/grates/blob/main/docs/source/_static/grates_banner.png)

What is grates?
---------------

grates is a free Open Source software package for analyzing time variable gravity field solutions.
It is tailored for data sets of the GRACE and GRACE-FO missions.

The features of grates are:

 * File I/O for common data formats (GFC files, GRACE-FO SDS Technical Notes, SINEX)
 * Basic arithmetic operations for sets of potential coefficients
 * Spatial filtering of gravity field solutions
 * Meridional transport from ocean bottom pressure fields
 * Kalman smoother for the determination of short-term gravity field variations
 * Isotropic and anisotropic harmonic kernels

 grates is currently in early development with frequent interface changes.

Installation
------------

The recommended way to install grates is in a [conda](https://docs.conda.io/en/latest/index.html) environment:
```
conda create -n grates_env
conda activate grates_env
```
Then, install all dependencies:
```
conda install numpy scipy cartopy netcdf4 numpydoc sphinx pyyaml
```
To install the current development version of the package, first clone the repository or download the zip archive.
In the root directory of the package (i.e. the directory containing the ``setup.py`` file), running
```
python -m pip install .
```
will install the package.
If you want to modify or extend the package, you can install it in develop mode by running
```
python -m pip install -e .
```
instead.

License
-------

grates a free Open Source software released under the GPL v3 license.
See [License](LICENSE) for details.
