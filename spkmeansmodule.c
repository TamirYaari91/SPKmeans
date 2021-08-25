#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <math.h>
#include "spkmeans.h"
/*
#include "kmeans.h"
*/
#include "kmeans2.h"

#define FUNC(_flag, _name, _docstring) { #_name, (PyCFunction)_name, _flag, PyDoc_STR(_docstring) }


static PyObject *mat_to_Python_mat(double **mat, int, int); /*not sure if needs to be static or not*/

static PyObject *kmeans2_py(int, int, int, PyObject *, PyObject *, int, int);

static PyObject *spkmeans_Python(char *, char *, int, int);

static PyObject *kmeans2(int, int, int, PyObject *,
                         PyObject *, int, int);

static PyObject *fit(PyObject *, PyObject *);

static PyObject *fit2(PyObject *, PyObject *);


static PyMethodDef capiMethods[] = {
        FUNC(METH_VARARGS, fit, "first part"),
        FUNC(METH_VARARGS, fit2, "second part"),
        {NULL, NULL, 0, NULL}
};


static struct PyModuleDef moduleDef = {
        PyModuleDef_HEAD_INIT, "myspkmeans", NULL, -1, capiMethods};


PyMODINIT_FUNC
PyInit_myspkmeans(void) {
    return PyModule_Create(&moduleDef);
}

/*PyMODINIT_FUNC
initmyspkmeans(void) {
    Py_InitModule3("myspkmeans",capiMethods,"bla");
}*/

static PyObject * spkmeans_Python(char *filename, char *goal, int k, int source) { /*source == 0 -> C | source == 1 -> Python*/
    int i, N, dim;
    int *N_dim;
    double *eigenvalues;
    double **data_points, **W, **D, **V, **U;
    eigen *eigen_items;
    PyObject *res;

    res = PyLong_FromLong(-1);
    N_dim = get_N_dim_from_file(filename);
    N = N_dim[0];
    dim = N_dim[1];

    if (k >= N) {
        printf("Invalid Input!");
        return res;
    }

    if (strcmp(goal, "wam") == 0) {
        data_points = get_mat_from_file(filename, N, dim);
        W = wam(data_points, N, dim);
        print_mat(W, N, N);
        free_mat(W);
        free(data_points);
        free(N_dim);
        return res;
    }

    if (strcmp(goal, "ddg") == 0) {
        data_points = get_mat_from_file(filename, N, dim);
        W = wam(data_points, N, dim);
        D = ddg(W, N);
        print_mat(D, N, N);
        free_mat(W);
        free_mat(D);
        free(data_points);
        free(N_dim);
        return res;
    }

    if (strcmp(goal, "lnorm") == 0) {
        data_points = get_mat_from_file(filename, N, dim);
        W = wam(data_points, N, dim);
        D = ddg(W, N);
        lnorm(W, D, N);
        print_mat(W, N, N);
        free_mat(W);
        free_mat(D);
        free(data_points);
        free(N_dim);
        return res;
    }

    if (strcmp(goal, "jacobi") == 0) {
        W = get_mat_from_file(filename, N, N);
        V = jacobi(W, N);
        eigenvalues = get_diag(W, N);
        print_row(eigenvalues, N);
        printf("\n");
        for (i = 0; i < N; i++) {
            double *eigenvector = get_ith_column(V, i, N);
            print_row(eigenvector, N);
            printf("\n");
            free(eigenvector);
        }
        free_mat(W);
        free_mat(V);
        free(eigenvalues);
        free(N_dim);
        return res;
    }

    if (strcmp(goal, "spk") == 0) {
        data_points = get_mat_from_file(filename, N, dim);
        W = wam(data_points, N, dim);
        D = ddg(W, N);
        lnorm(W, D, N);
        V = jacobi(W, N);
        eigenvalues = get_diag(W, N);
        eigen_items = calloc(N, sizeof(eigen));
        assert_eigen_arr(eigen_items);
        for (i = 0; i < N; i++) {
            double *eigenvector = get_ith_column(V, i, N);
            eigen item;
            item.value = eigenvalues[i];
            item.vector = eigenvector;
            eigen_items[i] = item;
        }
        mergeSort(eigen_items, 0, N - 1);
        if (k == 0) {
            k = eigen_gap(eigen_items, N);
        }
        U = gen_mat_k_eigenvectors(N, k, eigen_items);
        normalize_mat(U, N, k); /*T - which is U after it was normalized - is of size N*k*/

        free_mat(W);
        free_mat(D);
        free_mat(V);
        free(eigenvalues);
        for (i = 0; i < N; i++) {
            free(eigen_items[i].vector);
        }
        free(eigen_items);
        free(data_points);

        if (source) {
            res = mat_to_Python_mat(U, N, k);   /*converts U to Pyobject*/
            free_mat(U);
            free(N_dim);
            return res;
        } else {
            kmeans(U, k, N);
            free_mat(U);
            free(N_dim);
            return res;
        }

    } else { /* goal is not any of the valid options */
        printf("Invalid Input!");
        free(N_dim);
        return res;
    }
};

static PyObject *kmeans2(int k, int num_of_lines, int dim, PyObject *centroids_py,
                         PyObject *points_to_cluster_py, int centroids_length, int points_to_cluster_length) {

    double *centroids;
    double *points_to_cluster;
    int i, max_iter, changed, iters;
    PyObject *list;

    max_iter = 300;

    if (centroids_length < 0 || points_to_cluster_length < 0) {
        return NULL;
    }

    centroids = (double *) calloc(centroids_length, sizeof(double));
    assert(centroids != NULL && "Problem in allocating centroids memory");

    points_to_cluster = (double *) calloc(points_to_cluster_length, sizeof(double));
    assert(points_to_cluster != NULL && "Problem in allocating points_to_cluster memory");

    for (i = 0; i < centroids_length; i++) {
        PyObject *item;
        item = PyList_GetItem(centroids_py, i);
        centroids[i] = PyFloat_AsDouble(item);
    }
    for (i = 0; i < points_to_cluster_length; i++) {
        PyObject *item;
        item = PyList_GetItem(points_to_cluster_py, i);
        points_to_cluster[i] = PyFloat_AsDouble(item);
    }

    iters = 0;
    while (1) {
        for (i = 0; i < num_of_lines; ++i) {
            set_cluster2(i, k, points_to_cluster, centroids, dim);
        }
        changed = update_centroids2(k, num_of_lines, points_to_cluster, centroids, dim);
        iters++;
        if (changed == 0 || iters == max_iter) {
            break;
        }
    }

    list = PyList_New(centroids_length);
    for (i = 0; i < centroids_length; i++) {
        PyList_SetItem(list, i, PyFloat_FromDouble(centroids[i]));
    }
    free(centroids);
    free(points_to_cluster);
    return list;
}


static PyObject *kmeans2_py(int k, int num_of_lines, int dim, PyObject *centroids_py,
                            PyObject *points_to_cluster_py, int centroids_length, int points_to_cluster_length) {
    return kmeans2(k, num_of_lines, dim, centroids_py,
                   points_to_cluster_py, centroids_length, points_to_cluster_length);
};


static PyObject *mat_to_Python_mat(double **mat, int N, int dim) {
    Py_ssize_t i, j, rows, columns;
    PyObject *res;

    rows = N;
    columns = dim;
    res = PyList_New(N);

    for (i = 0; i < rows; i++) {
        PyObject *item = PyList_New(dim);
        for (j = 0; j < columns; j++)
            PyList_SET_ITEM(item, j, PyFloat_FromDouble(mat[i][j]));
        PyList_SET_ITEM(res, i, item);
    }
    return res;
};

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
    /*return Py_BuildValue("O",
                         spkmeans(filename, goal, k, 1));*/ /*  Py_BuildValue(...) returns a PyObject*  */
    return Py_BuildValue("O", spkmeans_Python(filename, goal, k, 1));
}

static PyObject *fit2(PyObject *self, PyObject *args) {
    int k;
    int num_of_lines;
    int dim;
    PyObject *centroids_py;
    PyObject *points_to_cluster_py;
    int centroids_length;
    int points_to_cluster_length;

    if (!PyArg_ParseTuple(args, "lllOOll:fit2", &k, &num_of_lines, &dim, &centroids_py,
                          &points_to_cluster_py,
                          &centroids_length, &points_to_cluster_length)) {
        return NULL;
    }

    return Py_BuildValue("O", kmeans2_py(k, num_of_lines, dim, centroids_py, points_to_cluster_py,
                                         centroids_length, points_to_cluster_length));
}

/*
 * This array tells Python what methods this module has.
 * We will use it in the next structure
 */


