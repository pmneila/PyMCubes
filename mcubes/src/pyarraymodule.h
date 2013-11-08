
#ifndef _EXTMODULE_H
#define _EXTMODULE_H

#include <Python.h>
#include <stdexcept>

#define PY_ARRAY_UNIQUE_SYMBOL mcubes_PyArray_API
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"

#include <boost/mpl/clear.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/map.hpp>
#include <boost/mpl/begin_end.hpp>

#include <complex>

namespace mpl = boost::mpl;

// Map C++ types -> numpy types.
typedef mpl::map<
      mpl::pair<bool,                      mpl::int_<NPY_BOOL> >
    , mpl::pair<char,                      mpl::int_<NPY_BYTE> >
    , mpl::pair<short,                     mpl::int_<NPY_SHORT> >
    , mpl::pair<int,                       mpl::int_<NPY_INT> >
    , mpl::pair<long,                      mpl::int_<NPY_LONG> >
    , mpl::pair<long long,                 mpl::int_<NPY_LONGLONG> >
    , mpl::pair<unsigned char,             mpl::int_<NPY_UBYTE> >
    , mpl::pair<unsigned short,            mpl::int_<NPY_USHORT> >
    , mpl::pair<unsigned int,              mpl::int_<NPY_UINT> >
    , mpl::pair<unsigned long,             mpl::int_<NPY_ULONG> >
    , mpl::pair<unsigned long long,        mpl::int_<NPY_ULONGLONG> >
    , mpl::pair<float,                     mpl::int_<NPY_FLOAT> >
    , mpl::pair<double,                    mpl::int_<NPY_DOUBLE> >
    , mpl::pair<long double,               mpl::int_<NPY_LONGDOUBLE> >
    , mpl::pair<std::complex<float>,       mpl::int_<NPY_CFLOAT> >
    , mpl::pair<std::complex<double>,      mpl::int_<NPY_CDOUBLE> >
    , mpl::pair<std::complex<long double>, mpl::int_<NPY_CLONGDOUBLE> >
    > numpy_typemap;

// Integer types vector.
// typedef mpl::vector<char, short, int, long, long long,
//                     unsigned char, unsigned short, unsigned int,
//                     unsigned long, unsigned long long> integer_types;
//typedef mpl::begin<integer_types>::type integer_types_begin;
//typedef mpl::end<integer_types>::type integer_types_end;
typedef mpl::vector<char, short, int, long, long long> signed_integer_types;
typedef mpl::begin<signed_integer_types>::type signed_integer_types_begin;
typedef mpl::end<signed_integer_types>::type signed_integer_types_end;

template<typename T>
T PyArray_SafeGet(const PyArrayObject* aobj, const npy_intp* indaux)
{
    // HORROR.
    npy_intp* ind = const_cast<npy_intp*>(indaux);
    void* ptr = PyArray_GetPtr(const_cast<PyArrayObject*>(aobj), ind);
    switch(PyArray_TYPE(aobj))
    {
    case PyArray_BOOL:
        return static_cast<T>(*reinterpret_cast<bool*>(ptr));
    case PyArray_BYTE:
        return static_cast<T>(*reinterpret_cast<char*>(ptr));
    case PyArray_SHORT:
        return static_cast<T>(*reinterpret_cast<short*>(ptr));
    case PyArray_INT:
        return static_cast<T>(*reinterpret_cast<int*>(ptr));
    case PyArray_LONG:
        return static_cast<T>(*reinterpret_cast<long*>(ptr));
    case PyArray_LONGLONG:
        return static_cast<T>(*reinterpret_cast<long long*>(ptr));
    case PyArray_UBYTE:
        return static_cast<T>(*reinterpret_cast<unsigned char*>(ptr));
    case PyArray_USHORT:
        return static_cast<T>(*reinterpret_cast<unsigned short*>(ptr));
    case PyArray_UINT:
        return static_cast<T>(*reinterpret_cast<unsigned int*>(ptr));
    case PyArray_ULONG:
        return static_cast<T>(*reinterpret_cast<unsigned long*>(ptr));
    case PyArray_ULONGLONG:
        return static_cast<T>(*reinterpret_cast<unsigned long long*>(ptr));
    case PyArray_FLOAT:
        return static_cast<T>(*reinterpret_cast<float*>(ptr));
    case PyArray_DOUBLE:
        return static_cast<T>(*reinterpret_cast<double*>(ptr));
    case PyArray_LONGDOUBLE:
        return static_cast<T>(*reinterpret_cast<long double*>(ptr));
    default:
        throw std::runtime_error("data type not supported");
    }
}

template<typename T>
T PyArray_SafeSet(PyArrayObject* aobj, const npy_intp* indaux, const T& value)
{
    // HORROR.
    npy_intp* ind = const_cast<npy_intp*>(indaux);
    void* ptr = PyArray_GetPtr(aobj, ind);
    switch(PyArray_TYPE(aobj))
    {
    case PyArray_BOOL:
        *reinterpret_cast<bool*>(ptr) = static_cast<bool>(value);
        break;
    case PyArray_BYTE:
        *reinterpret_cast<char*>(ptr) = static_cast<char>(value);
        break;
    case PyArray_SHORT:
        *reinterpret_cast<short*>(ptr) = static_cast<short>(value);
        break;
    case PyArray_INT:
        *reinterpret_cast<int*>(ptr) = static_cast<int>(value);
        break;
    case PyArray_LONG:
        *reinterpret_cast<long*>(ptr) = static_cast<long>(value);
        break;
    case PyArray_LONGLONG:
        *reinterpret_cast<long long*>(ptr) = static_cast<long long>(value);
        break;
    case PyArray_UBYTE:
        *reinterpret_cast<unsigned char*>(ptr) = static_cast<unsigned char>(value);
        break;
    case PyArray_USHORT:
        *reinterpret_cast<unsigned short*>(ptr) = static_cast<unsigned short>(value);
        break;
    case PyArray_UINT:
        *reinterpret_cast<unsigned int*>(ptr) = static_cast<unsigned int>(value);
        break;
    case PyArray_ULONG:
        *reinterpret_cast<unsigned long*>(ptr) = static_cast<unsigned long>(value);
        break;
    case PyArray_ULONGLONG:
        *reinterpret_cast<unsigned long long*>(ptr) = static_cast<unsigned long long>(value);
        break;
    case PyArray_FLOAT:
        *reinterpret_cast<float*>(ptr) = static_cast<float>(value);
        break;
    case PyArray_DOUBLE:
        *reinterpret_cast<double*>(ptr) = static_cast<double>(value);
        break;
    case PyArray_LONGDOUBLE:
        *reinterpret_cast<long double*>(ptr) = static_cast<long double>(value);
        break;
    default:
        throw std::runtime_error("data type not supported");
    }
}

#endif
