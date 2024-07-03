# -*- encoding: utf-8 -*-

import runpy

from setuptools import setup

from setuptools.extension import Extension

from Cython.Build import cythonize
import numpy


__version__ = runpy.run_path("mcubes/version.py")["__version__"]


def extensions():

    numpy_include_dir = numpy.get_include()

    mcubes_module = Extension(
        "mcubes._mcubes",
        [
            "mcubes/src/_mcubes.pyx",
            "mcubes/src/pywrapper.cpp",
            "mcubes/src/marchingcubes.cpp"
        ],
        language="c++",
        extra_compile_args=['-std=c++11', '-Wall'],
        include_dirs=[numpy_include_dir],
        depends=[
            "mcubes/src/marchingcubes.h",
            "mcubes/src/pyarray_symbol.h",
            "mcubes/src/pyarraymodule.h",
            "mcubes/src/pywrapper.h"
        ],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )

    return cythonize(
        [mcubes_module],
        language_level="3"
    )


setup(
    name="PyMCubes",
    version=__version__,
    description="Marching cubes for Python",
    author="Pablo Márquez Neila",
    author_email="pablo.marquez@unibe.ch",
    url="https://github.com/pmneila/PyMCubes",
    license="BSD 3-clause",
    long_description="""
    Marching cubes for Python
    """,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    packages=["mcubes"],
    ext_modules=extensions(),
    install_requires=['numpy>=1.21', 'scipy>=1.0.0'],
)
