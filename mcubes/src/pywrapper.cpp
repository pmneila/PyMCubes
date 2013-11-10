
#include "pywrapper.h"

#include "marchingcubes.h"

// struct PythonToCFunc
// {
//     py::object func;
//     PythonToCFunc(py::object func) {this->func = func;}
//     double operator()(double x, double y, double z)
//     {
//         return py::extract<double>(func(x,y,z));
//     }
// };

// py::object marching_cubes_func(py::object lower, py::object upper,
//     int numx, int numy, int numz, py::object f, double isovalue)
// {
//     std::vector<double> vertices;
//     std::vector<int> polygons;
    
//     // Copy the lower and upper coordinates to a C array.
//     double lower_[3];
//     double upper_[3];
//     for(int i=0; i<3; ++i)
//     {
//         lower_[i] = py::extract<double>(lower[i]);
//         upper_[i] = py::extract<double>(upper[i]);
//     }
    
//     // Marching cubes.
//     mc::marching_cubes(lower_, upper_, numx, numy, numz, PythonToCFunc(f), isovalue, vertices, polygons);
    
//     // Copy the result to two Python ndarrays.
//     npy_intp size_vertices = vertices.size();
//     npy_intp size_polygons = polygons.size();
//     PyArrayObject* verticesarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_vertices, PyArray_DOUBLE));
//     PyArrayObject* polygonsarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_polygons, PyArray_INT));
    
//     std::vector<double>::const_iterator it = vertices.begin();
//     for(int i=0; it!=vertices.end(); ++i, ++it)
//         *reinterpret_cast<double*>(PyArray_GETPTR1(verticesarr, i)) = *it;
//     std::vector<int>::const_iterator it2 = polygons.begin();
//     for(int i=0; it2!=polygons.end(); ++i, ++it2)
//         *reinterpret_cast<int*>(PyArray_GETPTR1(polygonsarr, i)) = *it2;
    
//     return py::make_tuple(verticesarr, polygonsarr);
// }

struct PyArrayToCFunc
{
    PyArrayObject* arr;
    PyArrayToCFunc(PyArrayObject* arr) {this->arr = arr;}
    double operator()(int x, int y, int z)
    {
        npy_intp c[3] = {x,y,z};
        return PyArray_SafeGet<double>(arr, c);
    }
};

PyObject* marching_cubes(PyArrayObject* arr, double isovalue)
{
    if(PyArray_NDIM(arr) != 3)
        throw std::runtime_error("Only three-dimensional arrays are supported.");
    
    // Prepare data.
    npy_intp* shape = PyArray_DIMS(arr);
    int lower[3] = {0,0,0};
    int upper[3] = {shape[0]-1, shape[1]-1, shape[2]-1};
    int numx = upper[0] - lower[0];
    int numy = upper[1] - lower[1];
    int numz = upper[2] - lower[2];
    std::vector<double> vertices;
    std::vector<int> polygons;
    
    // Marching cubes.
    mc::marching_cubes(lower, upper, numx, numy, numz, PyArrayToCFunc(arr), isovalue,
                        vertices, polygons);
    
    // Copy the result to two Python ndarrays.
    npy_intp size_vertices = vertices.size();
    npy_intp size_polygons = polygons.size();
    PyArrayObject* verticesarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_vertices, PyArray_DOUBLE));
    PyArrayObject* polygonsarr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &size_polygons, PyArray_INT));
    
    std::vector<double>::const_iterator it = vertices.begin();
    for(int i=0; it!=vertices.end(); ++i, ++it)
        *reinterpret_cast<double*>(PyArray_GETPTR1(verticesarr, i)) = *it;
    std::vector<int>::const_iterator it2 = polygons.begin();
    for(int i=0; it2!=polygons.end(); ++i, ++it2)
        *reinterpret_cast<int*>(PyArray_GETPTR1(polygonsarr, i)) = *it2;
    
    PyObject* res = Py_BuildValue("(O,O)", verticesarr, polygonsarr);
    Py_XDECREF(verticesarr);
    Py_XDECREF(polygonsarr);
    
    return res;
}
