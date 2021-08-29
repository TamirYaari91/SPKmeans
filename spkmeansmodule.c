#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <math.h>
#include "spkmeans.h"
#include "kmeans2.h"

#define FUNC(_flag, _name, _docstring) { #_name, (PyCFunction)_name, _flag, PyDoc_STR(_docstring) }


static PyObject *mat_to_Python_mat(double **mat, int, int);

static PyObject *spkmeans_Python(char *, char *, int);

static PyObject *kmeans2(int, int, int, PyObject *, PyObject *, int, int);

static PyObject *fit(PyObject *, PyObject *);

static PyObject *fit2(PyObject *, PyObject *);

static PyObject *spk_wrapper_Python(char*,int,int,int);



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

static PyObject * spkmeans_Python(char *filename, char *goal, int k) {
    /*Main logic - see C comments for more comments as the code is almost identical other than the returned object*/

    int N, dim;
    int *N_dim;
    PyObject *res;

    res = PyLong_FromLong(-1);
    N_dim = get_N_dim_from_file(filename);
    N = N_dim[0];
    dim = N_dim[1];

    if (k >= N) {
        printf("Invalid Input!");
        free(N_dim);
        return res;
    }

    if (strcmp(goal, "wam") == 0) {
        wam_wrapper(filename,N,dim);
    }

    else if (strcmp(goal, "ddg") == 0) {
        ddg_wrapper(filename,N,dim);
    }

    else if (strcmp(goal, "lnorm") == 0) {
        lnorm_wrapper(filename,N,dim);
    }

    else if (strcmp(goal, "jacobi") == 0) {
        jacobi_wrapper(filename,N);
    }

    else if (strcmp(goal, "spk") == 0) {
        res = spk_wrapper_Python(filename,N,dim,k);
        free(N_dim);
        return res;

    } else { /* goal is not any of the valid options */
        printf("Invalid Input!");
    }
    free(N_dim);
    return res;
};

static PyObject * spk_wrapper_Python(char* filename,int N,int dim, int k) {
    /*wrapper for goal == spk - gets data points from file and calculates wam according to given logic*/
    double **data_points, **W, **V, **D, **U;
    double *eigenvalues;
    int i;
    eigen *eigen_items; /*struct containing eigen value paired to its respective eigen vector*/
    PyObject * res;

    data_points = get_mat_from_file(filename, N, dim);
    W = wam(data_points, N, dim);
    D = ddg(W, N);
    lnorm(W, D, N);
    V = jacobi(W, N);

    eigenvalues = get_diag(W, N);
    eigen_items = calloc(N, sizeof(eigen));
    assert_eigen_arr(eigen_items);
    for (i = 0; i < N; i++) { /*loop creates eigen_items with the repsective value and vector*/
        double *eigenvector = get_ith_column(V, i, N);
        eigen item;
        item.value = eigenvalues[i];
        item.vector = eigenvector;
        eigen_items[i] = item;
    }
    mergeSort(eigen_items,0,N-1); /*sorting the eigen_items according to eigen values - using mergeSort since it
    is stable and O(nlogn) WC*/
    if (k == 0) {
        k = eigen_gap(eigen_items, N); /*if k == 0 then Eigengap Heuristic is used, as instructed*/
    }
    U = gen_mat_k_eigenvectors(N, k, eigen_items); /*creates U using the first k eigenvectors - u1,...,uk*/
    normalize_mat(U, N, k); /*T - which is U after it was normalized - is of size N*k*/
    free_mat(W);
    free_mat(D);
    free_mat(V);
    free(eigenvalues);
    for (i = 0; i < N; i++) {
        free(eigen_items[i].vector);
    }
    free(eigen_items);
    free_mat(data_points);
    res = mat_to_Python_mat(U, N, k);   /*converts T (still called U) to Pyobject*/
    free_mat(U);
    return res; /*Returns T as a Python object to be used in initial KMeans++ logic in Python*/
}

static PyObject *kmeans2(int k, int num_of_lines, int dim, PyObject *centroids_py,
                         PyObject *points_to_cluster_py, int centroids_length, int points_to_cluster_length) {
    /*KMeans algorithm taken as is from EX2*/

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

static PyObject *mat_to_Python_mat(double **mat, int N, int dim) {
    /*converts a C matrix of doubles to a matrix of PyObjects*/

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


static PyObject *fit(PyObject *self, PyObject *args) {
    /*gets filename, goal and k as an input and runs the main algorithm logic*/

    char *filename;
    char *goal;
    int k;

    if (!PyArg_ParseTuple(args, "ssi:fit", &filename, &goal, &k)) {
        return NULL;
    }

    return Py_BuildValue("O", spkmeans_Python(filename, goal, k));
}

static PyObject *fit2(PyObject *self, PyObject *args) {
    /*if goal == spk, T matrix was processed in Python with KMeans++ logic and output is used by kmeans*/

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

    return Py_BuildValue("O", kmeans2(k, num_of_lines, dim, centroids_py, points_to_cluster_py,
                                         centroids_length, points_to_cluster_length));
}