#ifndef UNTITLED10_KMEANS2_H
#define UNTITLED10_KMEANS2_H

/*This is basically KMeans from EX2 - to be used when SPKMeans is initiated from Python*/

double distance2(const double[], const double *, int, int);

void set_cluster2(int, int, double *, double *, int);

double *cluster_mean2(int, const int *, const double *, int, int);

int update_centroids2(int, int, double *, double *, int);

int equal2(const double *, const double *, int);


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
    assert_double_arr(distances);

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
    assert_double_arr(center);
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
        if (size > 0) {
            center[i] = val / size;
        }
        else {
            center[i] = 0.0;
        }
        size = 0;
        val = 0.0;
    }

    return center;
}

int update_centroids2(int k, int num_of_points, double *p2c, double *centroids, int dim) {
    int *c2p;
    int i, j, changed;
    c2p = (int *) calloc(k * num_of_points, sizeof(int));
    assert_int_arr(c2p);

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
