from setuptools import setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='grates',
    version='0.1',
    author='Andreas Kvas',
    description='A python package to compute mass transport from satellite gravimetry',
    install_requires=['numpy', 'scipy', 'netcdf4', 'numpydoc', 'cartopy', 'matplotlib'],
    packages=['grates'],
    package_data={'grates': ['data/ddk_normals.npz', 'data/loadLoveNumbers_Gegout97.txt']}
)
