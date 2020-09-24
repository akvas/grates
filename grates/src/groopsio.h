/***********************************************/
/**
* @file groopsio.h
*
* @brief Python module for GROOPS I/O.
*
* @author Andreas Kvas
* @date 2014-04-10
*
*/
/***********************************************/

#ifndef __GROOPSIO__
#define __GROOPSIO__

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/npy_3kcompat.h"

#include "files/fileMatrix.h"
#include "files/fileGriddedData.h"
#include "files/fileSphericalHarmonics.h"
#include "files/fileTimeSplinesGravityfield.h"
#include "files/fileSatellite.h"
#include "files/fileArcList.h"
#include "files/fileNormalEquation.h"
#include "files/filePolygon.h"

static PyObject *groopsError;

/**
 * @defgroup groopsio GROOPS/Python: Python interface
 *
 * Classes and functions for boost enabled usage of GROOPS classes
 * in Python.
 * @{
 */

/***** FUNCTIONS *******************************/

/** @brief Create 2D Python numeric array of specific size (FORTRAN order).
 *
 * @param rows number of rows
 * @param columns number of columns
 *
 * @return obj pointer to PyObject
 */
PyObject* createDoubleArray(UInt rows, UInt columns)
{
  npy_intp size[2];
  size[0] = rows;
  size[1] = columns;

  PyObject *pyObj = PyArray_ZEROS(2, size, NPY_DOUBLE, 1/*fortran order*/);

  return pyObj;
}

/** @brief GROOPS Matrix to Python Array
 *
 * Copies matrix field into a numeric array.
 * Triangular/Symmetric matrices are stored fully.
 *
 * @param M GROOPS Matrix
 *
 * @return obj pointer to PyObject
 */
PyObject* fromMatrix(const Matrix& M)
{
  if(M.getType() == Matrix::SYMMETRIC)
    fillSymmetric(M);
  else if(M.getType() == Matrix::TRIANGULAR)
    zeroUnusedTriangle(M);

  PyObject *pyObj = createDoubleArray(M.rows(), M.columns());
  memcpy(PyArray_DATA((PyArrayObject*)(pyObj)), M.field(), M.size()*sizeof(Double));

  return pyObj;
}

/** @brief Python Array to GROOPS Matrix
 *
 * Copies numeric array (assumed to be in FORTRAN order) into matrix field.
 *
 * @param pyObj pointer to PyObject
 * @param t Matrix type (GENERAL, SYMMETRIC, TRIANGULAR)
 * @param uplo Matrix triangle (UPPER, LOWER)
 *
 * @return M GROOPS Matrix
 */
Matrix fromPyObject(PyObject *pyObj, Matrix::Type t = Matrix::GENERAL,
                    Matrix::Uplo uplo = Matrix::UPPER)
{
  PyObject *pyArr = PyArray_FROM_OTF(pyObj, NPY_DOUBLE, NPY_ARRAY_IN_FARRAY);
  Int ndims = PyArray_NDIM((PyArrayObject*)pyArr);
  if(ndims<1 || ndims>2)
    throw(Exception("Expected one or two dimensional Numpy Array (got "+ndims%"%i"s+" dimensions)."));
  npy_intp* size = PyArray_DIMS((PyArrayObject*)pyArr);
  UInt rows = size[0];
  UInt columns = ndims == 1 ? 1 : size[1];

  Matrix M;
  if(t==Matrix::GENERAL)
    M = Matrix(rows, columns);
  else
    M = Matrix(rows, t, uplo);

  if(size[0]*size[1] != 0)
    memcpy(M.field(), PyArray_DATA((PyArrayObject*)pyArr), M.size()*sizeof(Double));

  Py_DecRef(pyArr);
  return M;
}

/** @brief Read GROOPS matrix from file
 *
 * Reads contents from a GROOPS matrix file and
 * copies them into a numeric array.
 * Triangular/Symmetric matrices are stored fully.
 *
 * @param fname File name
 *
 * @return v Matrix contents as numeric array
 */
static PyObject* loadmat(PyObject* /*self*/, PyObject *args)
{
  try
  {
    const char *s;
    if(!PyArg_ParseTuple(args, "s", &s))
      throw(Exception("Unable to parse arguments."));

    Matrix M;
    readFileMatrix(FileName(std::string(s)), M);
    PyObject* pyObj = fromMatrix(M);

    return pyObj;
  }
  catch(std::exception& e)
  {
    PyErr_SetString(groopsError, e.what());
    return NULL;
  }
}

/** @brief Save numeric array to GROOPS matrix file format
 *
 * @param fname File name
 * @param A numeric array
 * @param t matrix type
 * @param uplo save only upper/lower triangle
 */
static PyObject* savemat(PyObject* /*self*/, PyObject *args)
{
  try
  {
    const char *s;
    const char *type;
    const char *uplo;
    PyObject *pyObj;
    if(!PyArg_ParseTuple(args, "sOss", &s, &pyObj, &type, &uplo))
      throw(Exception("Unable to parse arguments."));

    Matrix::Type type_rep = Matrix::GENERAL;
    if(std::string(type) == std::string("symmetric"))
      type_rep = Matrix::SYMMETRIC;
    else if(std::string(type) == std::string("triangular"))
      type_rep = Matrix::TRIANGULAR;

    Matrix::Uplo uplo_rep = Matrix::UPPER;
    if(std::string(uplo) == std::string("lower"))
      uplo_rep = Matrix::LOWER;

    Matrix M = fromPyObject(pyObj, type_rep, uplo_rep);
    writeFileMatrix(FileName(std::string(s)), M);

    Py_RETURN_NONE;
  }
  catch(std::exception& e)
  {
    PyErr_SetString(groopsError, e.what());
    return NULL;
  }
}

/** @brief Read GROOPS grid from file
 *
 * Reads contents from a GROOPS grid file and
 * copies them into a dictionary of numeric arrays.
 *
 * @param fname File name
 *
 * @return grid Python dictionary
 */
static PyObject* loadgridrectangular(PyObject* /*self*/, PyObject *args)
{
  try
  {
    const char *s;
    if(!PyArg_ParseTuple(args, "s", &s))
      throw(Exception("Unable to parse arguments."));

    GriddedDataRectangular G;
    readFileGriddedData(FileName(std::string(s)), G);

    const UInt dataCount = G.values.size();

    PyObject *return_tuple = PyTuple_New(dataCount + 4); // data, lon, lat, a, f

    for(UInt k = 0; k < dataCount; k++)
      PyTuple_SetItem(return_tuple, k, fromMatrix(G.values.at(k)));

    Matrix lons(G.longitudes.size(), 1);
    for(UInt k = 0; k < G.longitudes.size(); k++)
      lons(k, 0) = G.longitudes.at(k);

    Matrix lats(G.latitudes.size(), 1);
    for(UInt k = 0; k < G.latitudes.size(); k++)
      lats(k, 0) = G.latitudes.at(k);

    PyTuple_SetItem(return_tuple, dataCount + 0, fromMatrix(lons));
    PyTuple_SetItem(return_tuple, dataCount + 1, fromMatrix(lats));
    PyTuple_SetItem(return_tuple, dataCount + 2, PyFloat_FromDouble(G.ellipsoid.a()));
    PyTuple_SetItem(return_tuple, dataCount + 3, PyFloat_FromDouble(1.0/G.ellipsoid.f()));

    return return_tuple;
  }
  catch(std::exception& e)
  {
    PyErr_SetString(groopsError, e.what());
    return NULL;
  }
}

/** @brief Read GROOPS grid from file
 *
 * Reads contents from a GROOPS grid file and
 * copies them into a dictionary of numeric arrays.
 *
 * @param fname File name
 *
 * @return grid Python dictionary
 */
static PyObject* loadgrid(PyObject* /*self*/, PyObject *args)
{
  try
  {
    const char *s;
    if(!PyArg_ParseTuple(args, "s", &s))
      throw(Exception("Unable to parse arguments."));

    GriddedData G;
    readFileGriddedData(FileName(std::string(s)), G);

    PyObject* data = createDoubleArray(G.points.size(), 4 + G.values.size()); // lon, lat, h, area, values
    {
      Angle L, B;
      Double h;

      for(UInt k=0; k<G.points.size(); k++)
      {
        G.ellipsoid(G.points.at(k), L, B, h);
        *(static_cast<Double*>(PyArray_GETPTR2((PyArrayObject*)data, k, 0))) = static_cast<Double>(L);
        *(static_cast<Double*>(PyArray_GETPTR2((PyArrayObject*)data, k, 1))) = static_cast<Double>(B);
        *(static_cast<Double*>(PyArray_GETPTR2((PyArrayObject*)data, k, 2))) = h;
        if(G.areas.size() == G.points.size()) *(static_cast<Double*>(PyArray_GETPTR2((PyArrayObject*)data, k, 3))) = G.areas.at(k);
        for(UInt l = 0; l<G.values.size(); l++)
          *(static_cast<Double*>(PyArray_GETPTR2((PyArrayObject*)data, k, 4+l))) = G.values.at(l).at(k);
      }
    }

    PyObject *return_tuple = PyTuple_New(3); // data, a, f
    PyTuple_SetItem(return_tuple, 0, data);
    PyTuple_SetItem(return_tuple, 1, PyFloat_FromDouble(G.ellipsoid.a()));
    PyTuple_SetItem(return_tuple, 2, PyFloat_FromDouble(1.0/G.ellipsoid.f()));

    return return_tuple;
  }
  catch(std::exception& e)
  {
    PyErr_SetString(groopsError, e.what());
    return NULL;
  }
}

/** @brief Save dictionary to GROOPS grid file format
 *
 * @param fname File name
 * @param grid Python dictionary
 */
static PyObject* savegrid(PyObject* /*self*/, PyObject* args)
{
  try
  {
    const char *s;
    Double a = 0.0;
    Double f = 0.0;
    PyObject *data_array;
    if(!PyArg_ParseTuple(args, "sOdd", &s, &data_array, &a, &f))
      throw(Exception("Unable to parse arguments."));

    Ellipsoid ell(a, f);
    Matrix data = fromPyObject(data_array);
    const UInt pointCount = data.rows();
    const UInt valueCount = data.columns() - 4;

    std::vector<Vector3d> _point(pointCount);
    std::vector<Double> _area(pointCount);
    std::vector< std::vector<Double> > _value(valueCount, std::vector<Double>(pointCount));

    for(UInt k = 0; k<pointCount; k++)
    {
      _point.at(k) = ell(Angle(data(k, 0)), Angle(data(k, 1)), Angle(data(k, 2)));
      _area.at(k) = data(k, 3);
      for(UInt l = 0; l<valueCount; l++)
        _value.at(l).at(k) = data(k, 4+l);
    }

    writeFileGriddedData(FileName(std::string(s)), GriddedData(ell, _point, _area, _value));

    Py_RETURN_NONE;
  }
  catch(std::exception& e)
  {
    PyErr_SetString(groopsError, e.what());
    return NULL;
  }
}

/** @brief Load spherical harmonic coefficients from gfc-file
 *
 * @param fname File name
 * @return gf Python dictionary
 */
static PyObject* loadgravityfield(PyObject* /*self*/, PyObject* args)
{
  try
  {
    const char *s;
    if(!PyArg_ParseTuple(args, "s", &s))
      throw(Exception("Unable to parse arguments."));

    SphericalHarmonics coeffs;
    readFileSphericalHarmonics(FileName(std::string(s)), coeffs);
    const Bool hasSigmas = (coeffs.sigma2cnm().size()) && ((quadsum(coeffs.sigma2cnm())+quadsum(coeffs.sigma2snm())) != 0);

    Matrix anm = coeffs.cnm();
    anm.setType(Matrix::GENERAL);
    axpy(1.0, coeffs.snm().slice(1, 1, coeffs.maxDegree(), coeffs.maxDegree()).trans(), anm.slice(0, 1, coeffs.maxDegree(), coeffs.maxDegree()));

    Matrix sigma2anm(coeffs.maxDegree()+1, coeffs.maxDegree()+1, NAN_EXPR);
    if(hasSigmas)
    {
      sigma2anm = coeffs.sigma2cnm();
      sigma2anm.setType(Matrix::GENERAL);
      axpy(1.0, coeffs.sigma2snm().slice(1, 1, coeffs.maxDegree(), coeffs.maxDegree()).trans(), sigma2anm.slice(0, 1, coeffs.maxDegree(), coeffs.maxDegree()));
    }

    PyObject *return_tuple = PyTuple_New(4); // GM, R, anm, sigma_anm

    PyTuple_SetItem(return_tuple, 0, PyFloat_FromDouble(coeffs.GM()));
    PyTuple_SetItem(return_tuple, 1, PyFloat_FromDouble(coeffs.R()));
    PyTuple_SetItem(return_tuple, 2, fromMatrix(anm));
    PyTuple_SetItem(return_tuple, 3, fromMatrix(sigma2anm));

    return return_tuple;
  }
  catch(std::exception& e)
  {
    PyErr_SetString(groopsError, e.what());
    return NULL;
  }
}

/** @brief Save Python dictionary to gfc-file
 *
 * @param fname File name
 * @param gf Python dictionary
 */
static PyObject* savegravityfield(PyObject* /*self*/, PyObject* args)
{
  try
  {
    const char *s;
    Double GM = 0.0;
    Double R = 0.0;
    Int hasSigmas = 0;
    PyObject *anm, *sigma2anm;
    if(!PyArg_ParseTuple(args, "sddOiO", &s, &GM, &R, &anm, &hasSigmas, &sigma2anm))
      throw(Exception("Unable to parse arguments."));

    Matrix _cnm = fromPyObject(anm);
    const UInt maxDegree = _cnm.rows()-1;
    Matrix _snm(maxDegree+1, Matrix::TRIANGULAR, Matrix::LOWER);
    copy(_cnm.slice(0, 1, maxDegree, maxDegree).trans(), _snm.slice(1, 1, maxDegree, maxDegree));
    zeroUnusedTriangle(_snm);
    _cnm.setType(Matrix::TRIANGULAR, Matrix::LOWER);
    zeroUnusedTriangle(_cnm);

    SphericalHarmonics harm;
    if(hasSigmas)
    {
      Matrix _sigma2cnm = fromPyObject(sigma2anm);
      Matrix _sigma2snm(maxDegree+1, Matrix::TRIANGULAR, Matrix::LOWER);
      copy(_sigma2cnm.slice(0, 1, maxDegree, maxDegree).trans(), _sigma2snm.slice(1, 1, maxDegree, maxDegree));
      zeroUnusedTriangle(_snm);
      _sigma2cnm.setType(Matrix::TRIANGULAR, Matrix::LOWER);
      zeroUnusedTriangle(_sigma2cnm);

      harm = SphericalHarmonics(GM, R, _cnm, _snm, _sigma2cnm, _sigma2snm);
    }
    else
      harm = SphericalHarmonics(GM, R, _cnm, _snm);

    writeFileSphericalHarmonics(FileName(std::string(s)), harm);

    Py_RETURN_NONE;
  }
  catch(std::exception& e)
  {
    PyErr_SetString(groopsError, e.what());
    return NULL;
  }
}

/** @brief Load spherical harmonic coefficients from TimeSplinesFile
 *
 * @param fname File name
 * @param t point in time to evaluate spline time series
 * @return gf Python dictionary
 */
static PyObject* loadtimesplines(PyObject* /*self*/, PyObject* args)
{
  try
  {
    Double mjd = 0.0;
    const char *s;
    if(!PyArg_ParseTuple(args, "sd", &s, &mjd))
      throw(Exception("Unable to parse arguments."));
    std::string fname(s);

    TimeSplinesGravityfieldFile timeSplinesFile;
    timeSplinesFile.open(FileName(fname));

    Time t = mjd2time(mjd);
    SphericalHarmonics coeffs = timeSplinesFile.sphericalHarmonics(t);
    timeSplinesFile.close();

    Matrix anm = coeffs.cnm();
    axpy(1.0, coeffs.snm().slice(1, 1, coeffs.maxDegree(), coeffs.maxDegree()).trans(), anm.slice(0, 1, coeffs.maxDegree(), coeffs.maxDegree()));

    PyObject *return_tuple = PyTuple_New(3);
    PyTuple_SetItem(return_tuple, 0, PyFloat_FromDouble(coeffs.GM()));
    PyTuple_SetItem(return_tuple, 1, PyFloat_FromDouble(coeffs.R()));
    PyTuple_SetItem(return_tuple, 2, fromMatrix(anm));

    return return_tuple;
  }
  catch(std::exception& e)
  {
    PyErr_SetString(groopsError, e.what());
    return NULL;
  }
}

/** @brief Load arcs from instrument files
 *
 * @param fname File name
 *
 * @return tuple Python tuple
 */
static PyObject* loadinstrument(PyObject* /*self*/, PyObject* args)
{
  try
  {
    const char *s;
    if(!PyArg_ParseTuple(args, "s", &s))
      throw(Exception("Unable to parse arguments."));
    std::string fname(s);

    InstrumentFile fileInstrument;
    fileInstrument.open(FileName(fname));

    PyObject* return_values = PyTuple_New(2); // (time data), EpochType

    PyObject* arc_list = PyTuple_New(fileInstrument.arcCount());
    Epoch::Type type = Epoch::EMPTY;

    for(UInt arcNo = 0; arcNo<fileInstrument.arcCount(); arcNo++)
    {
      Arc arc = fileInstrument.readArc(arcNo);

      Matrix M;
      if(arc.size())
      {
        M = arc.matrix();
        type = arc.at(0).getType();
      }

      PyTuple_SetItem(arc_list, arcNo, fromMatrix(M));
    }

    PyTuple_SetItem(return_values, 0, arc_list);
    PyTuple_SetItem(return_values, 1, Py_BuildValue("i", static_cast<Int>(type)));

    return return_values;
  }
  catch(std::exception& e)
  {
    PyErr_SetString(groopsError, e.what());
    return NULL;
  }
}

  /** @brief Load arcs from instrument files
 *
 * @param fname File name
 *
 * @return tuple Python tuple
 */
static PyObject* loadstarcamera(PyObject* /*self*/, PyObject* args)
{
  try
  {
    const char *s;
    if(!PyArg_ParseTuple(args, "s", &s))
      throw(Exception("Unable to parse arguments."));
    std::string fname(s);

    StarCameraArc starCameraArc = InstrumentFile::read(FileName(fname));

    PyObject* return_values = PyTuple_New(2); // time, data

    PyObject* data = PyTuple_New(starCameraArc.size());

    Vector mjd(starCameraArc.size());
    for(UInt k = 0; k<starCameraArc.size(); k++)
    {
      Time epoch = starCameraArc.at(k).time;
      Rotary3d rot = starCameraArc.at(k).rotary;

      PyTuple_SetItem(data, k, fromMatrix(rot.matrix()));
      mjd(k) = epoch.mjd();
    }

    PyTuple_SetItem(return_values, 0, fromMatrix(mjd));
    PyTuple_SetItem(return_values, 1, data);

    return return_values;
  }
  catch(std::exception& e)
  {
    PyErr_SetString(groopsError, e.what());
    return NULL;
  }
}

/** @brief Save a list of numpy arrays as instrument file
 *
 * @param fname File name
 * @param instrument tuple of numeric arrays
 * @param epochType integer representation of @ref Epoch::Type
 *
 */
static PyObject* saveinstrument(PyObject* /*self*/, PyObject* args)
{
  try
  {
    const char *s;
    PyObject* list;
    Int epochType = 0;
    if(!PyArg_ParseTuple(args, "sOi", &s, &list, &epochType))
      throw(Exception("Unable to parse arguments."));
    std::string fname(s);

    UInt arcCount = PyList_Size(list);
    Epoch::Type type = static_cast<Epoch::Type>(epochType);

    std::vector<Arc> arcList(arcCount);
    for(UInt arcNo = 0; arcNo<arcCount; arcNo++)
    {
      Matrix M = fromPyObject(PyList_GetItem(list, arcNo));
      arcList[arcNo] = Arc(M, type);
    }

    InstrumentFile fileInstrument;
    fileInstrument.write(FileName(fname), arcList);

    Py_RETURN_NONE;
  }
  catch(std::exception& e)
  {
    PyErr_SetString(groopsError, e.what());
    return NULL;
  }
}

/** @brief Load a GROOPS arc list
 *
 * @param fname File name
 * @return PyObject* tuple (mjd, arcIntervals)
 *
 */
static PyObject* loadarclist(PyObject* /*self*/, PyObject* args)
{
  try
  {
    const char *s;
    if(!PyArg_ParseTuple(args, "s", &s))
      throw(Exception("Unable to parse arguments."));
    std::string fname(s);

    std::vector<UInt> arcsInterval;
    std::vector<Time> timesInterval;

    readFileArcList(FileName(fname), arcsInterval, timesInterval);

    PyObject* list = PyTuple_New(2);
    PyObject* aI = PyTuple_New(arcsInterval.size());
    PyObject* tI = PyTuple_New(timesInterval.size());

    PyTuple_SetItem(list, 0, aI);
    PyTuple_SetItem(list, 1, tI);

    for(UInt k = 0; k<arcsInterval.size(); k++)
    {
      PyTuple_SetItem(aI, k, Py_BuildValue("i", arcsInterval.at(k)));
      PyTuple_SetItem(tI, k, Py_BuildValue("d", timesInterval.at(k).mjd()));
    }

    return list;
  }
  catch(std::exception& e)
  {
    PyErr_SetString(groopsError, e.what());
    return NULL;
  }
}

  /** @brief Load GROOPS normal equation info
 *
 * @param fname File name
 * @return PyObject* tuple (b, a)
 *
 */
static PyObject* loadnormalsinfo(PyObject* /*self*/, PyObject* args)
{
  try
  {
    const char *s;
    if(!PyArg_ParseTuple(args, "s", &s))
      throw(Exception("Unable to parse arguments."));

    Matrix n;
    NormalEquationInfo info;
    readFileNormalEquation(FileName(std::string(s)), info, n);

    PyObject* parameter_names = PyTuple_New(info.parameterName.size());
    for(UInt k = 0; k<info.parameterName.size(); k++)
      PyTuple_SetItem(parameter_names, k, Py_BuildValue("s", info.parameterName.at(k).str().c_str()));

    Vector blockIndex(info.blockIndex.size());
    for(UInt k = 0; k<info.blockIndex.size(); k++)
      blockIndex(k) = info.blockIndex.at(k);

    PyObject* list = PyTuple_New(5);
    PyTuple_SetItem(list, 0, fromMatrix(info.lPl));
    PyTuple_SetItem(list, 1, Py_BuildValue("i", info.observationCount));
    PyTuple_SetItem(list, 2, parameter_names);
    PyTuple_SetItem(list, 3, fromMatrix(blockIndex));
    PyTuple_SetItem(list, 4, fromMatrix(info.usedBlocks));

    return list;
  }
  catch(std::exception& e)
  {
    PyErr_SetString(groopsError, e.what());
    return NULL;
  }
}

/** @brief Load GROOPS normal equation
 *
 * @param fname File name
 * @return PyObject* tuple (b, a)
 *
 */
static PyObject* loadnormals(PyObject* /*self*/, PyObject* args)
{
  try
  {
    const char *s;
    if(!PyArg_ParseTuple(args, "s", &s))
      throw(Exception("Unable to parse arguments."));

    Matrix N, n;
    NormalEquationInfo info;
    readFileNormalEquation(FileName(std::string(s)), info, N, n);

    PyObject* list = PyTuple_New(4);
    PyTuple_SetItem(list, 0, fromMatrix(N));
    PyTuple_SetItem(list, 1, fromMatrix(n));
    PyTuple_SetItem(list, 2, fromMatrix(info.lPl));
    PyTuple_SetItem(list, 3, Py_BuildValue("i", info.observationCount));

    return list;
  }
  catch(std::exception& e)
  {
    PyErr_SetString(groopsError, e.what());
    return NULL;
  }
}

/** @brief Save matrix and metadata to GROOPS normals file format
 *
 * @param fname File name
 * @param grid Python dictionary
 */
static PyObject* savenormals(PyObject* /*self*/, PyObject* args)
{
  try
  {
    const char *s;
    PyObject *matrix;
    PyObject *rhs;
    PyObject *lPl;
    int obsCount = 0;
    if(!PyArg_ParseTuple(args, "sOOOi", &s, &matrix, &rhs, &lPl, &obsCount))
      throw(Exception("Unable to parse arguments."));

    NormalEquationInfo info;

    Matrix N = fromPyObject(matrix, Matrix::SYMMETRIC, Matrix::UPPER);
    Matrix n = fromPyObject(rhs);
    info.lPl = fromPyObject(lPl);
    info.observationCount = obsCount;
    info.parameterName.resize(n.rows());
    info.blockIndex = std::vector<UInt>({0, n.rows()});
    info.usedBlocks = Matrix(1, 1, 1.0);

    writeFileNormalEquation(FileName(std::string(s)), info, N, n);

    Py_RETURN_NONE;
  }
  catch(std::exception& e)
  {
    PyErr_SetString(groopsError, e.what());
    return NULL;
  }
}

  /** @brief Read a polygon list in GROOPS file format
 *
 * @param fname File name
 */
static PyObject* loadpolygon(PyObject* /*self*/, PyObject* args)
{
  try
  {
    const char *s;
    if(!PyArg_ParseTuple(args, "s", &s))
      throw(Exception("Unable to parse arguments."));

    std::vector<Polygon> poly;
    readFilePolygon(FileName(std::string(s)), poly);

    PyObject* polygons = PyTuple_New(poly.size());
    for(UInt k = 0; k < poly.size(); k++)
    {
      Matrix data(poly.at(k).L.size(), 2);
      copy(poly.at(k).L, data.column(0));
      copy(poly.at(k).B, data.column(1));
      PyTuple_SetItem(polygons, k, fromMatrix(data));
    }

    return polygons;
  }
  catch(std::exception& e)
  {
    PyErr_SetString(groopsError, e.what());
    return NULL;
  }
}
// @} group groopsio

#endif
