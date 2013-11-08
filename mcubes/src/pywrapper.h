
#ifndef _PYWRAPPER_H
#define _PYWRAPPER_H

#include <Python.h>
#include "pyarraymodule.h"

PyObject* marching_cubes(PyArrayObject* arr, double isovalue);

#endif // _PYWRAPPER_H
