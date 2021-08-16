#ifndef UNTITLED10_KMEANS2_H
#define UNTITLED10_KMEANS2_H

double distance2(const double[], const double *, int, int);

void set_cluster2(int, int, double *, double *, int);

double *cluster_mean2(int, const int *, const double *, int, int);

int update_centroids2(int, int, double *, double *, int);

int equal2(const double *, const double *, int);

static PyObject *kmeans2(int, int, int, PyObject *, PyObject *, int, int);

/*
static PyObject *fit2(PyObject *self, PyObject *args);
*/

/*
#define FUNC(_flag, _name, _docstring) { #_name, (PyCFunction)_name, _flag, PyDoc_STR(_docstring) }
*/

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

double distance2(const double *p, const double *centroids, int cluster, int dim) {
    double d;
    int i;

    d = 0;
    for (i = 0; i < dim; ++i) {
        double multi;
        multi = (p[i] - *((centroids + (cluster * dim) + i)));
        d += multi * multi;
    }
    return d;
}

void set_cluster2(int p_index, int k, double *point_to_cluster, double *centroids, int dim) {
    int i;
    int min_index;
    double *distances;

    min_index = 0;
    distances = (double *) calloc(k, sizeof(double));
    assert(distances);

    for (i = 0; i < k; ++i) {
        distances[i] = distance2((point_to_cluster + p_index * (dim + 1)), centroids, i, dim);
        if (distances[i] < distances[min_index]) {
            min_index = i;
        }
    }
    point_to_cluster[(p_index * (dim + 1) + dim)] = min_index;
}

double *cluster_mean2(int cluster, const int *c2p, const double *p2c, int dim, int num_of_points) {
    int size;
    double val;
    double p_val;
    int i, j;
    static double *center;
    center = (double *) calloc(dim, sizeof(double));
    assert(center);
    size = 0;
    val = 0.0;

    for (i = 0; i < dim; ++i) {
        for (j = 0; j < num_of_points; ++j) {

            int p_index = *(c2p + (cluster * num_of_points) + j);
            if (p_index < 0) {
                break;
            }
            size++;
            p_val = *(p2c + p_index * (dim + 1) + i);
            val += p_val;
        }
        center[i] = val / size;
        size = 0;
        val = 0.0;
    }

    return center;
}

int update_centroids2(int k, int num_of_points, double *p2c, double *centroids, int dim) {
    int *c2p;
    int i, j, changed;
    c2p = (int *) calloc(k * num_of_points, sizeof(int));
    assert(c2p);

    for (i = 0; i < k; ++i) {
        for (j = 0; j < num_of_points; ++j) {
            c2p[i * num_of_points + j] = -1;
        }
    }
    for (i = 0; i < num_of_points; ++i) {
        int cluster = p2c[i * (dim + 1) + dim];
        for (j = 0; j < num_of_points; ++j) {
            if (c2p[(cluster * num_of_points) + j] == -1) {
                c2p[(cluster * num_of_points) + j] = i;

                break;
            }
        }
    }
    changed = 0;
    for (i = 0; i < k; ++i) {
        double *new_centroid = cluster_mean2(i, c2p, p2c, dim, num_of_points);
        if (equal2((centroids + dim * i), new_centroid, dim) == 0) {
            for (j = 0; j < dim; ++j) {
                *(centroids + dim * i + j) = new_centroid[j];
            }
            changed = 1;
        }
        free(new_centroid);
    }
    free(c2p);
    return changed;
}

int equal2(const double *arr1, const double *arr2, int dim) {
    int i;
    for (i = 0; i < dim; ++i) {
        if (arr1[i] != arr2[i]) {
            return 0;
        }
    }
    return 1;
}

#endif
