/***********************************************/
/**
* @file groopsio.cpp
*
* @brief Ppython module for GROOPS I/O.
*
* @author Andreas Kvas
* @date 2014-04-10
*
*/
/***********************************************/

#include "groopsio.h"

/**
 * @brief Ppython module for GROOPS I/O.
 */
extern "C" {

  static PyMethodDef iobase_methods[] = {
    {"loadmat", (PyCFunction)loadmat, METH_VARARGS, ""},
    {"savemat", (PyCFunction)savemat, METH_VARARGS, ""},

    {"loadgridrectangular", (PyCFunction)loadgridrectangular, METH_VARARGS, ""},
    {"loadgrid", (PyCFunction)loadgrid, METH_VARARGS, ""},
    {"savegrid", (PyCFunction)savegrid, METH_VARARGS, ""},

    {"loadgravityfield", (PyCFunction)loadgravityfield, METH_VARARGS, ""},
    {"savegravityfield", (PyCFunction)savegravityfield, METH_VARARGS, ""},

    {"loadtimesplines", (PyCFunction)loadtimesplines, METH_VARARGS, ""},

    {"loadarclist", (PyCFunction)loadarclist, METH_VARARGS, ""},

    {"loadinstrument", (PyCFunction)loadinstrument, METH_VARARGS, ""},
    {"saveinstrument", (PyCFunction)saveinstrument, METH_VARARGS, ""},

    {"loadstarcamera", (PyCFunction)loadstarcamera, METH_VARARGS, ""},

    {"loadnormalsinfo", (PyCFunction)loadnormalsinfo, METH_VARARGS, ""},
    {"loadnormals", (PyCFunction)loadnormals, METH_VARARGS, ""},
    {"savenormals", (PyCFunction)savenormals, METH_VARARGS, ""},

    {"loadpolygon", (PyCFunction)loadpolygon, METH_VARARGS, ""},

    {NULL, NULL, 0, NULL}
  };

  static struct PyModuleDef iobase_definition = {
    PyModuleDef_HEAD_INIT, /*m_base*/
    "groopsiobase",        /*m_name*/
    NULL,                  /*m_doc*/
    -1,                    /*m_size*/
    iobase_methods,        /*m_methods*/
    NULL,                  /*m_slots*/
    NULL,                  /*m_traverse*/
    NULL,                  /*m_clear*/
    NULL                   /*m_free*/
  };

  PyMODINIT_FUNC PyInit_groopsiobase()
  {
    PyObject *module = PyModule_Create(&iobase_definition);
    import_array();

    groopsError = PyErr_NewException("groopsiobase.GROOPSError", NULL, NULL);
    Py_INCREF(groopsError);
    PyModule_AddObject(module, "GROOPSError", groopsError);

    return module;
  }
}
