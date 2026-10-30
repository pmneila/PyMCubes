#include "pywrapper.h"

#include "marchingcubes.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <array>
#include <vector>

namespace
{

// Releases the GIL for as long as it is in scope, so that the marching cubes
// passes can run on other threads and other Python threads can make progress.
class gil_release
{
public:
    gil_release() : state_(PyEval_SaveThread()) {}
    ~gil_release() { PyEval_RestoreThread(state_); }

    gil_release(const gil_release&) = delete;
    gil_release& operator=(const gil_release&) = delete;

private:
    PyThreadState* state_;
};

// Reads the volume without touching the Python C API, so it is safe to call
// from several threads at once.  Requires native byte order and an aligned
// buffer; marching_cubes() converts anything else before reaching this point.
template<typename value_type>
class array_value_at
{
public:
    explicit array_value_at(PyArrayObject* arr)
        : data_(static_cast<const char*>(PyArray_DATA(arr)))
    {
        const npy_intp* strides = PyArray_STRIDES(arr);
        stride_[0] = strides[0];
        stride_[1] = strides[1];
        stride_[2] = strides[2];
    }

    double operator()(size_t i, size_t j, size_t k) const
    {
        const npy_intp offset = static_cast<npy_intp>(i) * stride_[0]
            + static_cast<npy_intp>(j) * stride_[1]
            + static_cast<npy_intp>(k) * stride_[2];
        return static_cast<double>(*reinterpret_cast<const value_type*>(data_ + offset));
    }

private:
    const char* data_;
    npy_intp stride_[3];
};

// Owns a reference; used for the converted copy of a byte-swapped or unaligned
// input array.
class array_ref
{
public:
    explicit array_ref(PyArrayObject* obj = NULL) : obj_(obj) {}
    ~array_ref() { Py_XDECREF(obj_); }

    array_ref(const array_ref&) = delete;
    array_ref& operator=(const array_ref&) = delete;

    PyArrayObject* get() const { return obj_; }

private:
    PyArrayObject* obj_;
};

PyObject* build_result(const std::vector<double>& vertices,
    const std::vector<size_t>& polygons)
{
    npy_intp size_vertices = static_cast<npy_intp>(vertices.size());
    npy_intp size_polygons = static_cast<npy_intp>(polygons.size());
    PyArrayObject* verticesarr = reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNew(1, &size_vertices, NPY_DOUBLE));
    PyArrayObject* polygonsarr = reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNew(1, &size_polygons, NPY_ULONG));
    if(verticesarr == NULL || polygonsarr == NULL)
    {
        Py_XDECREF(verticesarr);
        Py_XDECREF(polygonsarr);
        return NULL;
    }

    if(!vertices.empty())
        std::memcpy(PyArray_DATA(verticesarr), vertices.data(),
                    vertices.size() * sizeof(double));
    unsigned long* polygons_out =
        static_cast<unsigned long*>(PyArray_DATA(polygonsarr));
    for(size_t i=0; i<polygons.size(); ++i)
        polygons_out[i] = static_cast<unsigned long>(polygons[i]);

    PyObject* res = Py_BuildValue("(O,O)", verticesarr, polygonsarr);
    Py_XDECREF(verticesarr);
    Py_XDECREF(polygonsarr);
    return res;
}

} // namespace


PyObject* marching_cubes_func(PyObject* lower, PyObject* upper,
    int numx, int numy, int numz, PyObject* pyfunc, double isovalue,
    int num_threads)
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

        // PyFloat_AsDouble returns -1.0 on failure, which is also a valid
        // coordinate, so the error indicator is what decides.
        lower_[i] = PyFloat_AsDouble(l);
        if(PyErr_Occurred())
        {
            Py_DECREF(l);
            Py_DECREF(u);
            throw std::runtime_error("lower coordinates must be numbers");
        }
        upper_[i] = PyFloat_AsDouble(u);
        if(PyErr_Occurred())
        {
            Py_DECREF(l);
            Py_DECREF(u);
            throw std::runtime_error("upper coordinates must be numbers");
        }

        Py_DECREF(l);
        Py_DECREF(u);
    }

    if(numx < 2 || numy < 2 || numz < 2)
        throw std::runtime_error("numx, numy and numz must be at least 2.");

    // Sample the field at every grid node. Each node is visited once (and only
    // once), which lets the marching cubes core run without the GIL. The
    // spacing must stay bit-identical to what mc::marching_cubes() recomputes
    // from (upper - lower) / (n - 1), because the sampled values are attached
    // to those node positions when a vertex is interpolated.
    const double dx = (upper_[0] - lower_[0]) / (numx - 1);
    const double dy = (upper_[1] - lower_[1]) / (numy - 1);
    const double dz = (upper_[2] - lower_[2]) / (numz - 1);

    const size_t nx = static_cast<size_t>(numx);
    const size_t ny = static_cast<size_t>(numy);
    const size_t nz = static_cast<size_t>(numz);
    if(nx > (std::numeric_limits<size_t>::max() / ny) / nz)
        throw std::runtime_error("the requested grid is too large");
    // 8 bytes per node on top of the core's own buffers (128 MB at 256^3).
    std::vector<double> grid(nx * ny * nz);

    for(int i=0; i<numx; ++i)
    {
        const double x = lower_[0] + dx*i;
        for(int j=0; j<numy; ++j)
        {
            const double y = lower_[1] + dy*j;
            for(int k=0; k<numz; ++k)
            {
                const double z = lower_[2] + dz*k;

                PyObject* res = PyObject_CallFunction(pyfunc, "(d,d,d)", x, y, z);
                if(res == NULL)
                {
                    // Return NULL so that Cython re-raises the exception raised
                    // by the callback instead of building a partial surface.
                    return NULL;
                }
                const double value = PyFloat_AsDouble(res);
                Py_DECREF(res);
                if(PyErr_Occurred())
                    return NULL;
                grid[(static_cast<size_t>(i) * ny + static_cast<size_t>(j)) * nz
                    + static_cast<size_t>(k)] = value;
            }
        }
    }

    auto grid_value_at = [&](size_t i, size_t j, size_t k) -> double {
        return grid[(i * ny + j) * nz + k];
    };

    {
        gil_release release;
        mc::marching_cubes(lower_, upper_, numx, numy, numz, grid_value_at,
            isovalue, vertices, polygons, num_threads);
    }

    return build_result(vertices, polygons);
}


template<typename value_type>
PyObject* marching_cubes_typed(PyArrayObject* arr, double isovalue, int num_threads)
{
    npy_intp* shape = PyArray_DIMS(arr);
    std::array<npy_intp, 3> lower{0, 0, 0};
    std::array<npy_intp, 3> upper{shape[0]-1, shape[1]-1, shape[2]-1};
    int numx = static_cast<int>(upper[0] - lower[0] + 1);
    int numy = static_cast<int>(upper[1] - lower[1] + 1);
    int numz = static_cast<int>(upper[2] - lower[2] + 1);
    std::vector<double> vertices;
    std::vector<size_t> polygons;

    array_value_at<value_type> value_at(arr);

    {
        gil_release release;
        mc::marching_cubes(lower, upper, numx, numy, numz, value_at, isovalue,
            vertices, polygons, num_threads);
    }

    return build_result(vertices, polygons);
}


PyObject* marching_cubes(PyArrayObject* arr, double isovalue, int num_threads)
{
    if(PyArray_NDIM(arr) != 3)
        throw std::runtime_error("Only three-dimensional arrays are supported.");

    const npy_intp* shape = PyArray_DIMS(arr);
    for(int d=0; d<3; ++d)
    {
        if(shape[d] > std::numeric_limits<int>::max())
            throw std::runtime_error("array dimension is too large");
    }

    if(!PyArray_ISNOTSWAPPED(arr) || !PyArray_ISALIGNED(arr))
    {
        // The reader below dereferences the buffer as a native value_type, so a
        // byte-swapped or unaligned input gets a native, aligned double copy.
        array_ref native(reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
            reinterpret_cast<PyObject*>(arr), NPY_DOUBLE,
            NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED)));
        if(native.get() == NULL)
            return NULL;
        return marching_cubes_typed<npy_double>(native.get(), isovalue, num_threads);
    }

    switch(PyArray_TYPE(arr))
    {    case NPY_BOOL:
        return marching_cubes_typed<npy_bool>(arr, isovalue, num_threads);
    case NPY_BYTE:
        return marching_cubes_typed<npy_byte>(arr, isovalue, num_threads);
    case NPY_SHORT:
        return marching_cubes_typed<npy_short>(arr, isovalue, num_threads);
    case NPY_INT:
        return marching_cubes_typed<npy_int>(arr, isovalue, num_threads);
    case NPY_LONG:
        return marching_cubes_typed<npy_long>(arr, isovalue, num_threads);
    case NPY_LONGLONG:
        return marching_cubes_typed<npy_longlong>(arr, isovalue, num_threads);
    case NPY_UBYTE:
        return marching_cubes_typed<npy_ubyte>(arr, isovalue, num_threads);
    case NPY_USHORT:
        return marching_cubes_typed<npy_ushort>(arr, isovalue, num_threads);
    case NPY_UINT:
        return marching_cubes_typed<npy_uint>(arr, isovalue, num_threads);
    case NPY_ULONG:
        return marching_cubes_typed<npy_ulong>(arr, isovalue, num_threads);
    case NPY_ULONGLONG:
        return marching_cubes_typed<npy_ulonglong>(arr, isovalue, num_threads);
    case NPY_FLOAT:
        return marching_cubes_typed<npy_float>(arr, isovalue, num_threads);
    case NPY_DOUBLE:
        return marching_cubes_typed<npy_double>(arr, isovalue, num_threads);
    case NPY_LONGDOUBLE:
        return marching_cubes_typed<npy_longdouble>(arr, isovalue, num_threads);
    default:
        break;
    }

    // Any other dtype that numpy can safely cast (float16, and whatever gets
    // added later) takes the same native double copy as the byte-swapped and
    // unaligned inputs above, so support does not depend on the memory layout.
    // Dtypes that cannot be cast safely (complex, object, strings) make
    // PyArray_FROM_OTF raise a TypeError.
    array_ref converted(reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(
        reinterpret_cast<PyObject*>(arr), NPY_DOUBLE,
        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED)));
    if(converted.get() == NULL)
        return NULL;
    return marching_cubes_typed<npy_double>(converted.get(), isovalue, num_threads);
}
