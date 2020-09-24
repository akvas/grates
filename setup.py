import setuptools
from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup
from numpy.distutils.system_info import get_info
import os


def configuration(parent_package='', top_path=None):

    groops_dir = os.getenv('GROOPS_SOURCE_DIR')
    if groops_dir is None:
        groops_dir = os.path.join(os.path.expanduser('~'), 'groops', 'source')

    include_dirs = [groops_dir]
    library_dirs = []
    libraries = ['expat', 'z', 'gfortran', 'stdc++fs']

    lapack_opts = get_info('lapack_opt', 0)
    include_dirs.extend(lapack_opts['include_dirs'])
    library_dirs.extend(lapack_opts['library_dirs'])
    libraries.extend(lapack_opts['libraries'])

    source_files = []
    with open('grates/src/sources.list', 'r') as f:
        for line in f:
            if len(line) > 0 and not line.startswith('#'):
                source_files.append(os.path.join(groops_dir, line.strip()))

    config = Configuration()
    config.add_installed_library('groopsdeps', source_files, 'grates/src/lib',
                                 build_info={'include_dirs': include_dirs, 'libraries': libraries,
                                             'library_dirs': library_dirs})

    libraries.append('groopsdeps')
    library_dirs.append('grates/src/lib')
    config.add_extension('groopsiobase',  ["grates/src/groopsio.cpp"], language='c++',
                        include_dirs=include_dirs, libraries=libraries, library_dirs=library_dirs)

    return config


with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='grates',
    version='0.1',
    author='Andreas Kvas',
    description='A python package to compute mass transport from satellite gravimetry',
    install_requires=['numpy', 'scipy', 'netcdf4', 'numpydoc', 'cartopy'],
    packages=['grates'],
    configuration=configuration,
    package_data={'grates': ['data/ddk_normals.npz', 'data/loadLoveNumbers_Gegout97.txt']}
)
