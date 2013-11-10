# -*- encoding: utf-8 -*-

from distutils.core import setup
#from distutils.core import setup
from Cython.Build import cythonize

import numpy

# Get the version number.
numpy_include_dir = numpy.get_include()

mcubes_module = cythonize('mcubes/src/_mcubes.pyx')

mcubes_module[0].include_dirs.append(numpy_include_dir)
mcubes_module[0].sources.append("mcubes/src/pywrapper.cpp")
mcubes_module[0].sources.append("mcubes/src/marchingcubes.cpp")

setup(name="PyMCubes",
    version="0.0.1",
    description="Marching cubes for Python",
    author="Pablo Márquez Neila",
    author_email="p.mneila@upm.es",
    url="",
    license="GPL",
    long_description="""
    Marching cubes for Python
    """,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    packages=["mcubes"],
    ext_modules=mcubes_module
    )
