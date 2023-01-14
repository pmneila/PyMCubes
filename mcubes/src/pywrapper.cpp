
#include "pywrapper.h"

#include "marchingcubes.h"

#include <stdexcept>
#include <array>


PyObject* marching_cubes_func(PyObject* lower, PyObject* upper,
    int numx, int numy, int numz, PyObject* pyfunc, double isovalue)
{
    std::vector<double> vertices;
    std::vector<size_t> polygons;

    // Copy the lower and upper coordinates to a C array.
    std::array<double,3> lower_;
    std::array<double,3> upper_;
    for(int i=0; i<3; ++i)
    {
        PyObject* l = PySequence_GetItem(lower, i);
        if(l == NULL)
            throw std::runtime_error("len(lower) < 3");
        PyObject* u = PySequence_GetItem(upper, i);
        if(u == NULL)
        {
            Py_DECREF(l);
            throw std::runtime_error("len(upper) < 3");
        }

        lower_[i] = PyFloat_AsDouble(l);
        upper_[i] = PyFloat_AsDouble(u);

        Py_DECREF(l);
        Py_DECREF(u);
        if(lower_[i]==-1.0 || upper_[i]==-1.0)
        {
            if(PyErr_Occurred())
                throw std::runtime_error("unknown error");
        }
    }

    auto pyfunc_to_cfunc = [&](double x, double y, double z) -> double {
        PyObject* res = PyObject_CallFunction(pyfunc, "(d,d,d)", x, y, z);
        if(res == NULL)
            return 0.0;

        double result = PyFloat_AsDouble(res);
        Py_DECREF(res);
        return result;
    };

    // Marching cubes.
    mc::marching_cubes(lower_, upper_, numx, numy, numz, pyfunc_to_cfunc, isovalue, vertices, polygons);

    // Copy the result to two Python ndarrays.
    npy_intp size_vertices = vertices.size();
    npy_intp size_polygons = polygons.size();
    PyArrayObject* verticesarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_vertices, NPY_DOUBLE));
    PyArrayObject* polygonsarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_polygons, NPY_ULONG));

    std::vector<double>::const_iterator it = vertices.begin();
    for(int i=0; it!=vertices.end(); ++i, ++it)
        *reinterpret_cast<double*>(PyArray_GETPTR1(verticesarr, i)) = *it;
    std::vector<size_t>::const_iterator it2 = polygons.begin();
    for(int i=0; it2!=polygons.end(); ++i, ++it2)
        *reinterpret_cast<unsigned long*>(PyArray_GETPTR1(polygonsarr, i)) = *it2;

    PyObject* res = Py_BuildValue("(O,O)", verticesarr, polygonsarr);
    Py_XDECREF(verticesarr);
    Py_XDECREF(polygonsarr);
    return res;
}


PyObject* marching_cubes(PyArrayObject* arr, double isovalue)
{
    if(PyArray_NDIM(arr) != 3)
        throw std::runtime_error("Only three-dimensional arrays are supported.");

    // Prepare data.
    npy_intp* shape = PyArray_DIMS(arr);
    std::array<long, 3> lower{0, 0, 0};
    std::array<long, 3> upper{shape[0]-1, shape[1]-1, shape[2]-1};
    long numx = upper[0] - lower[0] + 1;
    long numy = upper[1] - lower[1] + 1;
    long numz = upper[2] - lower[2] + 1;
    std::vector<double> vertices;
    std::vector<size_t> polygons;

    auto pyarray_to_cfunc = [&](long x, long y, long z) -> double {
        const npy_intp c[3] = {x, y, z};
        return PyArray_SafeGet<double>(arr, c);
    };

    // Marching cubes.
    mc::marching_cubes(lower, upper, numx, numy, numz, pyarray_to_cfunc, isovalue,
                        vertices, polygons);

    // Copy the result to two Python ndarrays.
    npy_intp size_vertices = vertices.size();
    npy_intp size_polygons = polygons.size();
    PyArrayObject* verticesarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_vertices, NPY_DOUBLE));
    PyArrayObject* polygonsarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_polygons, NPY_ULONG));

    std::vector<double>::const_iterator it = vertices.begin();
    for(int i=0; it!=vertices.end(); ++i, ++it)
        *reinterpret_cast<double*>(PyArray_GETPTR1(verticesarr, i)) = *it;
    std::vector<size_t>::const_iterator it2 = polygons.begin();
    for(int i=0; it2!=polygons.end(); ++i, ++it2)
        *reinterpret_cast<unsigned long*>(PyArray_GETPTR1(polygonsarr, i)) = *it2;

    PyObject* res = Py_BuildValue("(O,O)", verticesarr, polygonsarr);
    Py_XDECREF(verticesarr);
    Py_XDECREF(polygonsarr);

    return res;
}
