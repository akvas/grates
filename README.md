![grates logo](https://github.com/akvas/grates/blob/main/docs/source/_static/grates_banner.png)

What is grates?
---------------

grates is a free Open Source software package for analyzing time variable gravity field solutions.
It is tailored for data sets of the GRACE and GRACE-FO missions.

The features of l3py are:

 * File I/O for common data formats (GFC files, GRACE-FO SDS Technical Notes, SINEX)
 * Basic arithmetic operations for sets of potential coefficients
 * Spatial filtering of gravity field solutions
 * Meridional transport from ocean bottom pressure fields
 * Kalman smoother for the determination of short-term gravity field variations

 grates is currently in early development with frequent interface changes.

Installation
------------

To install the current development version of the package, first clone the repository or download the zip archive.
In the root directory
of the package (i.e. the directory containing the ``setup.py`` file), running

    pip install .

will install the package and its dependencies.

License
-------

grates a free Open Source software released under the GPL v3 license.
See [License](LICENSE) for details.
