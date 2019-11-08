# -*- encoding: utf-8 -*-

from setuptools import setup

from setuptools.extension import Extension


class lazy_cythonize(list):
    """
    Lazy evaluate extension definition, to allow correct requirements install.
    """
    
    def __init__(self, callback):
        super(lazy_cythonize, self).__init__()
        self._list, self.callback = None, callback
    
    def c_list(self):
        if self._list is None:
            self._list = self.callback()
        
        return self._list
    
    def __iter__(self):
        for e in self.c_list():
            yield e
    
    def __getitem__(self, ii):
        return self.c_list()[ii]
    
    def __len__(self):
        return len(self.c_list())


def extensions():
    
    from Cython.Build import cythonize
    import numpy
    
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
    )
    
    return cythonize([mcubes_module])

setup(
    name="PyMCubes",
    version="0.0.11",
    description="Marching cubes for Python",
    author="Pablo Márquez Neila",
    author_email="pablo.marquez@artorg.unibe.ch",
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
    ext_modules=lazy_cythonize(extensions),
    requires=['numpy', 'Cython', 'PyCollada'],
    setup_requires=['numpy', 'Cython']
)
