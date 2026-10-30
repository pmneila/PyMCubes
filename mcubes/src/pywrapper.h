
#ifndef _PYWRAPPER_H
#define _PYWRAPPER_H

#include <Python.h>
#include "pyarraymodule.h"

#include <vector>

// Both entry points take num_threads: <= 0 lets the core pick one thread per
// CPU core, 1 keeps the computation single-threaded. The Python layer
// additionally rejects negative values.
PyObject* marching_cubes(PyArrayObject* arr, double isovalue, int num_threads);
PyObject* marching_cubes_func(PyObject* lower, PyObject* upper,
    int numx, int numy, int numz, PyObject* f, double isovalue, int num_threads);

#endif // _PYWRAPPER_H
