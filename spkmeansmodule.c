#define PY_SSIZE_T_CLEAN  /* For all # variants of unit formats (s#, y#, etc.) use Py_ssize_t rather than int. */

/*#include <Python.h>*/       /* MUST include <Python.h>, this implies inclusion of the following standard headers:
                             <stdio.h>, <string.h>, <errno.h>, <limits.h>, <assert.h> and <stdlib.h> (if available). */
#include <math.h>         /* include <Python.h> has to be before any standard headers are included */
#include "spkmeans.h"
#include <python3.7/Python.h>
/*
 * Helper function that will not be exposed (meaning, should be static)
 */

/*
 * A geometric series up to n. sum_up_to_n(z^n)
 */



/*
 * This actually defines the geo function using a wrapper C API function
 * The wrapping function needs a PyObject* self argument.
 * This is a requirement for all functions and methods in the C API.
 * It has input PyObject *args from Python.
 */
static PyObject *fit(PyObject *self, PyObject *args) {
    char *filename;
    char *goal;
    int k;
    /* This parses the Python arguments into a double (d)  variable named z and int (i) variable named n*/
    if (!PyArg_ParseTuple(args, "ssi:fit", &filename, &goal, &k)) {
        return NULL; /* In the CPython API, a NULL value is never valid for a
                        PyObject* so it is used to signal that an error has occurred. */
    }

/* This builds the answer ("d" = Convert a C double to a Python floating point number) back into a python object */
    return Py_BuildValue("O",
                         spkmeans(filename, goal, k, 1)); /*  Py_BuildValue(...) returns a PyObject*  */
}

/*
 * This array tells Python what methods this module has.
 * We will use it in the next structure
 */
static PyMethodDef capiMethods[] = {
        {"fit",                   /* the Python method name that will be used */
         (PyCFunction) fit, /* the C-function that implements the Python function and returns static PyObject*  */
         METH_VARARGS,           /* flags indicating parametersaccepted for this function */
         PyDoc_STR("A C function to calculate spkmeans")}, /*  The docstring for the function */
        { NULL, NULL, 0, NULL }     /* The last entry must be all NULL as shown to act as a
                                 sentinel. Python looks for this entry to know that all
                                 of the functions for the module have been defined. */
};


static struct PyModuleDef moduleDef = {
        PyModuleDef_HEAD_INIT, "myspkmeans", NULL, -1, capiMethods};


/*
 * The PyModuleDef structure, in turn, must be passed to the interpreter in the module’s initialization function.
 * The initialization function must be named PyInit_name(), where name is the name of the module and should match
 * what we wrote in struct PyModuleDef.
 * This should be the only non-static item defined in the module file
 */
PyMODINIT_FUNC
PyInit_myspkmeans(void) {
    return PyModule_Create(&moduleDef);
}


