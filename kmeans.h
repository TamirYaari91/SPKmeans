#ifndef UNTITLED10_KMEANS_H
#define UNTITLED10_KMEANS_H

double distance(const double[], const double *, int, int);

void set_cluster(int, int, double *, double *);

double *cluster_mean(int, const int *, const double *, int, int);

int update_centroids(int, int, double *, double *);

int equal(const double *, const double *, int);

int kmeans(double **T, int k, int N) {
    int changed;
    double *centroids;
    double *points_to_cluster;
    int i, j, iters, max_iter = 300;

    centroids = (double *) malloc(k * (k + 1) * sizeof(double));
    assert(centroids);
    points_to_cluster = (double *) malloc(N * (k + 1) * sizeof(double));
    assert(points_to_cluster);
    for (i = 0; i < N * (k + 1); ++i) {
        points_to_cluster[i] = 0.0;
    }

    int cnt = 0;
    for (i = 0; i < N; i++) {
        for (j = 0; j < k + 1; j++) {
            if (cnt == k) {
                points_to_cluster[(i * (k + 1)) + j] = -1;
                cnt = 0;
            } else {
                points_to_cluster[(i * (k + 1)) + j] = T[i][j];
                cnt++;
            }
        }
    }

    j = 0;
    for (i = 1; i < k * (k + 1) + 1; ++i) {
        if (i % (k + 1) == 0) {
            continue;
        }
        centroids[j] = points_to_cluster[i - 1];
        j++;
    }
    iters = 0;
    while (1) {
        for (i = 0; i < N; ++i) {
            set_cluster(i, k, points_to_cluster, centroids);
        }
        changed = update_centroids(k, N, points_to_cluster, centroids);
        iters++;
        if (changed == 0 || iters == max_iter) {
            break;
        }
    }

    for (i = 0; i < (k * k); ++i) {
        printf("%.4f", centroids[i]);
        if ((i + 1) % k != 0) {
            printf(",");
        } else {
            if (i < (k * k) - 1) {
                printf("\n");
            }
        }
    }
    free(centroids);
    free(points_to_cluster);
    return 0;
}

double distance(const double *p, const double *centroids, int cluster, int k) {
    double d = 0;
    int i;
    for (i = 0; i < k; ++i) {
        double multi;
        multi = (p[i] - *((centroids + cluster * k) + i));
        d += multi * multi;
    }
    return d;
}

void set_cluster(int p_index, int k, double *point_to_cluster, double *centroids) {
    int i;
    int min_index = 0;
    double *distances;
    distances = (double *) malloc(k * sizeof(double));
    assert(distances);

    for (i = 0; i < k; ++i) {
        distances[i] = distance((point_to_cluster + p_index * (k + 1)), centroids, i, k);
        if (distances[i] < distances[min_index]) {
            min_index = i;
        }
    }
    point_to_cluster[(p_index * (k + 1) + k)] = min_index;
}

double *cluster_mean(int cluster, const int *c2p, const double *p2c, int k, int num_of_points) {
    int size;
    double val;
    double p_val;
    int i, j;
    static double *center;
    center = (double *) malloc(k * sizeof(double));
    assert(center);
    size = 0;
    val = 0.0;
    for (i = 0; i < k; ++i) {
        for (j = 0; j < num_of_points; ++j) {
            int p_index = *(c2p + cluster * num_of_points + j);
            if (p_index < 0) {
                break;
            }
            size++;
            p_val = *(p2c + p_index * (k + 1) + i);
            val += p_val;
        }
        center[i] = val / size;
        size = 0;
        val = 0.0;
    }
    return center;
}

int update_centroids(int k, int num_of_points, double *p2c, double *centroids) {
    int *c2p;
    int i, j, changed;
    c2p = (int *) malloc(k * num_of_points * sizeof(int));
    assert(c2p);

    for (i = 0; i < k; ++i) {
        for (j = 0; j < num_of_points; ++j) {
            c2p[i * num_of_points + j] = -1;
        }
    }

    for (i = 0; i < num_of_points; ++i) {
        int cluster = p2c[i * (k + 1) + k];
        for (j = 0; j < num_of_points; ++j) {
            if (c2p[(cluster * num_of_points) + j] == -1) {
                c2p[(cluster * num_of_points) + j] = i;

                break;
            }
        }
    }
    changed = 0;
    for (i = 0; i < k; ++i) {
        double *new_centroid = cluster_mean(i, c2p, p2c, k, num_of_points);
        if (equal((centroids + k * i), new_centroid, k) == 0) {
            for (j = 0; j < k; ++j) {
                *(centroids + k * i + j) = new_centroid[j];
            }
            changed = 1;
        }
        free(new_centroid);
    }
    free(c2p);
    return changed;
}

int equal(const double *arr1, const double *arr2, int k) {
    int i;
    for (i = 0; i < k; ++i) {
        if (arr1[i] != arr2[i]) {
            return 0;
        }
    }
    return 1;
}

#endif //UNTITLED10_KMEANS_H
