#define  _GNU_SOURCE

/*
#include <Python.h>
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include "spkmeans.h"
#include "kmeans.h"
/*
#include "kmeans2.h"
*/

/*
#include <python3.7/Python.h>
*/

#define eps pow(10,-15)

/*
What's left?
- comment the code with explanations
- check in nova - make sure to change Python include in all relevant files!
- compare to testers in Whatsapp group?
- free N_dim done in all places?
- splitting spkmeans to before and after N_dim to only go over file twice instead of 3 times
*/



int main(int argc, char *argv[]) {
    int k;
    char *end_k;

    if (argc < 3) {
        printf("Invalid Input!");
        return 0;
    }

    k = (int) strtol(argv[1], &end_k, 10);
    if (*end_k != '\0' || k < 0) {
        printf("Invalid Input!");
        return 0;
    }

    spkmeans_C(argv[3], argv[2], k, 0);
    return 0;
}

double **spkmeans_C(char *filename, char *goal, int k, int source) { /*source == 0 -> C | source == 1 -> Python*/
    int i, N, dim;
    int *N_dim;
    double *eigenvalues;
    double **data_points, **W, **D, **V, **U;
    eigen *eigen_items;

    N_dim = get_N_dim_from_file(filename);
    N = N_dim[0];
    dim = N_dim[1];

    if (k >= N) {
        printf("Invalid Input!");
        exit(1);
    }

    if (strcmp(goal, "wam") == 0) {
        data_points = get_mat_from_file(filename, N, dim);
        W = wam(data_points, N, dim);
        print_mat(W, N, N);
/*
        free_mat(W);
*/
        free(data_points);
        free(N_dim);
        return W;
    }

    if (strcmp(goal, "ddg") == 0) {
        data_points = get_mat_from_file(filename, N, dim);
        W = wam(data_points, N, dim);
        D = ddg(W, N);
        print_mat(D, N, N);
        free_mat(W);
/*
        free_mat(D);
*/
        free(data_points);
        free(N_dim);
        return D;
    }

    if (strcmp(goal, "lnorm") == 0) {
        data_points = get_mat_from_file(filename, N, dim);
        W = wam(data_points, N, dim);
        D = ddg(W, N);
        lnorm(W, D, N);
        print_mat(W, N, N);
/*
        free_mat(W);
*/
        free_mat(D);
        free(data_points);
        free(N_dim);
        return W;
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
/*
        free_mat(V);
*/
        free(eigenvalues);
        free(N_dim);
        return V;
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
        mergeSort(eigen_items,0,N-1);
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
            /*res = mat_to_Python_mat(U, N, k);*/   /*converts U to Pyobject*/
/*
            free_mat(U);
*/
            free(N_dim);
            return U;
        } else {
            kmeans(U, k, N);
/*
            free_mat(U);
*/
            free(N_dim);
            return U;
        }

    } else { /* goal is not any of the valid options */
        printf("Invalid Input!");
        free(N_dim);
        exit(0);
    }
}

double norm(double *p1, double *p2, int dim) {
    int i;
    double res, sum;
    sum = 0;
    for (i = 0; i < dim; i++) {
        sum += pow(p1[i] - p2[i], 2);
    }
    res = sqrt(sum);
    return res;
}

double **wam(double **data_points, int N, int dim) {
    int i, j;
    double **W;
    double *block;

    j = 0;
    block = calloc(N * N, sizeof(double));
    assert_double_arr(block);
    W = calloc(N, sizeof(double *));
    assert_double_mat(W);
    for (i = 0; i < N; i++) {
        W[i] = block + i * N;
        /*W[i][i] = 0.0; */
        /*W_ii = 0 - not needed because calloc initializes to zero?*/
    }
    for (i = 0; i < N; i++) {
        while (j < N) {
            if (i != j) {
                W[i][j] = exp(-(norm(data_points[i], data_points[j], dim) / 2));
                W[j][i] = W[i][j];
            }
            j++;
        }
        j = i + 1;
    }
    return W;
}

/*void print_mat(double **mat, int N, int dim) {
    int row, columns;
    for (row = 0; row < N; row++) {
        for (columns = 0; columns < dim; columns++) {
            printf("%.4lf", mat[row][columns]);
            if (row < N - 1) {
                if (columns == dim - 1) {
                    printf("\n");
                } else {
                    printf(",");
                }
            } else {
                if (columns != dim - 1) {
                    printf(",");
                }
            }
        }
    }
}*/

void print_mat(double **mat, int N, int dim) {
    int row, columns;
    for (row = 0; row < N; row++) {
        for (columns = 0; columns < dim; columns++) {
            printf("%.4f", mat[row][columns]);
            if (columns == dim - 1) {
                printf("\n");
            } else {
                printf(",");
            }
        }
    }
}


void print_row(double *row, int len) {
    int i;
    for (i = 0; i < len; i++) {
        printf("%.4f", row[i]);
        if (i != len - 1) {
            printf(",");
        }
    }
}

double **ddg(double **wam_mat, int N) {
    int i, j;
    double **D;
    double *block;

    block = calloc(N * N, sizeof(double));
    assert_double_arr(block);
    D = calloc(N, sizeof(double *));
    assert_double_mat(D);
    for (i = 0; i < N; i++) {
        D[i] = block + i * N;
        for (j = 0; j < N; j++) {
            if (i == j) {
                D[i][j] = sum_row(wam_mat[i], N);
            }
            /*                        else {
                                        D[i][j] = 0.0;
                                    } not needed because calloc initializes to zero?*/
        }
    }
    return D;
}

void lnorm(double **W, double **D, int N) {
    diag_mat_pow_half(D, N);
    diag_mat_multi_reg_mat(D, W, N);
    reg_mat_multi_diag_mat(D, W, N);
    identity_minus_reg_mat(W, N);

}

double sum_row(const double *mat, int m) {
    int i;
    double res;
    res = 0;
    for (i = 0; i < m; i++) {
        res += mat[i];
    }
    return res;
}

void diag_mat_pow_half(double **mat, int N) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (i == j) {
                mat[i][j] = 1 / (sqrt(mat[i][j]));
            }
        }
    }
}

void diag_mat_multi_reg_mat(double **D, double **W, int N) { /*result mat is W - the reg mat - the second mat!*/
    int i, j;
    double d_ii;

    for (i = 0; i < N; i++) {
        d_ii = D[i][i];
        for (j = 0; j < N; j++) {
            W[i][j] *= d_ii;
        }
    }
}

void reg_mat_multi_diag_mat(double **D, double **W, int N) { /*result mat is W - the reg mat - the second mat!*/
    int i, j;
    double *D_diag;
    D_diag = calloc(N, sizeof(double));
    assert_double_arr(D_diag);
    for (i = 0; i < N; i++) {
        D_diag[i] = D[i][i];
    }
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            W[i][j] *= D_diag[j];
        }
    }
    free(D_diag);
}

void identity_minus_reg_mat(double **mat, int N) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (i == j) {
                mat[i][j] = 1 - mat[i][j];
            } else {
                mat[i][j] *= -1;
            }
        }
    }
}

void A_to_A_tag(double **A, double **V, int N) {
    int i, j, r;
    int *arr_max;
    double theta, s, t, c, a_ri, a_rj, a_ii, a_jj;
    /*
        double **P;
    */

    arr_max = max_indices_off_diag(A, N);
    i = arr_max[0];
    j = arr_max[1];
    free(arr_max);
    theta = (A[j][j] - A[i][i]) / (2 * A[i][j]);
    t = sign(theta) / (fabs(theta) + sqrt((pow(theta, 2)) + 1));
    c = 1 / sqrt((pow(t, 2)) + 1);
    s = t * c;
    /*    P = gen_P(s, c, i, j, N);
        multi_mat(V, P, N);
        free_mat(P);*/
    V_multi_P(V,s,c,N,i,j);

    for (r = 0; r < N; r++) {
        if ((r != j) && (r != i)) {
            a_ri = c * A[r][i] - s * A[r][j];
            a_rj = c * A[r][j] + s * A[r][i];
            A[r][i] = a_ri;
            A[r][j] = a_rj;
            A[j][r] = a_rj;
            A[i][r] = a_ri;
        }
    }
    a_ii = pow(c, 2) * A[i][i] + pow(s, 2) * A[j][j] - 2 * s * c * A[i][j];
    a_jj = pow(c, 2) * A[j][j] + pow(s, 2) * A[i][i] + 2 * s * c * A[i][j];
    A[j][j] = a_jj;
    A[i][i] = a_ii;
    A[i][j] = 0.0;
    A[j][i] = 0.0;
}

int *max_indices_off_diag(double **A, int N) {
    double val;
    int i, j, max_i, max_j;
    int *arr;

    val = -1;
    max_i = 0;
    max_j = 0;

    arr = calloc(2, sizeof(int));
    assert_int_arr(arr);

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (i != j) {
                if (fabs(A[i][j]) > val) {
                    val = fabs(A[i][j]);
                    max_i = i;
                    max_j = j;
                }
            }
        }
    }
    arr[0] = max_i;
    arr[1] = max_j;
    return arr;
}

int sign(double num) {
    if (num < 0) {
        return -1;
    } else {
        return 1;
    }
}

double off(double **A, int N) {
    int i, j;
    double res;

    res = 0;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (i != j) {
                res += pow(A[i][j], 2);
            }
        }
    }
    return res;
}

double **gen_id_mat(int N) {
    int i, j;
    double **I;
    double *block;

    block = calloc(N * N, sizeof(double));
    assert_double_arr(block);
    I = calloc(N, sizeof(double *));
    assert_double_mat(I);
    for (i = 0; i < N; i++) {
        I[i] = block + i * N;
        for (j = 0; j < N; j++) {
            if (i == j) {
                I[i][j] = 1.0;
            }
            /*                        else {
                                        I[i][j] = 0.0;
                                    } not needed because calloc initializes to zero?*/
        }
    }
    return I;
}

double **gen_mat(int N, int k) {
    int i;
    /*
            int j;
    */
    double **M;
    double *block;

    block = calloc(N * k, sizeof(double));
    assert_double_arr(block);
    M = calloc(N, sizeof(double *));
    assert_double_mat(M);
    for (i = 0; i < N; i++) {
        M[i] = block + i * k;
        /*
                        for (j = 0; j < k; j++) {
                            M[i][j] = 0.0;
                        } not needed because calloc initializes to zero?
        */
    }
    return M;
}


double **jacobi(double **A, int N) {
    int iter, max_iter;
    double diff;
    double **V;

    iter = 0;
    max_iter = 100;
    V = gen_id_mat(N);
    diff = 10;

    while (diff > eps && iter < max_iter) {
        double off_A = off(A, N);
        iter++;
        A_to_A_tag(A, V, N);
        diff = off_A - off(A, N);
    }
    return V;
}

/*double **gen_P(double s, double c, int i, int j, int N) {
    double **P;
    P = gen_id_mat(N);
    P[i][j] = s;
    P[j][i] = -s;
    P[i][i] = c;
    P[j][j] = c;
    return P;
}*/

/*void multi_mat(double **mat1, double **mat2, int N) {
    int i, j, k;
    double **res;
    res = gen_id_mat(N);
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            res[i][j] = 0;
            for (k = 0; k < N; k++)
                res[i][j] += mat1[i][k] * mat2[k][j];
        }
    }
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            mat1[i][j] = res[i][j];
        }
    }
    free_mat(res);
}*/

double *get_diag(double **mat, int N) {
    int i;
    double *diag;
    diag = calloc(N, sizeof(double));
    assert_double_arr(diag);

    for (i = 0; i < N; i++) {
        diag[i] = mat[i][i];
    }
    return diag;
}

double *get_ith_column(double **mat, int col_ind, int N) {
    int i;
    double *col;
    col = calloc(N, sizeof(double));
    assert_double_arr(col);
    for (i = 0; i < N; i++) {
        col[i] = mat[i][col_ind];
    }
    return col;
}

/*
 A function to implement merge sort, modified to sort eigen_items - Programiz.com
*/


/* Merge two subarrays L and M into arr */
void merge(eigen * arr, int p, int q, int r) {
    int i,j,k,n1,n2;
    eigen *L, *M;


    /* Create L ← A[p..q] and M ← A[q+1..r] */
    n1 = q - p + 1;
    n2 = r - q;

    L = calloc(n1,sizeof(eigen));
    assert_eigen_arr(L);
    M = calloc(n2,sizeof(eigen));
    assert_eigen_arr(M);

    for (i = 0; i < n1; i++)
        L[i] = arr[p + i];
    for (j = 0; j < n2; j++)
        M[j] = arr[q + 1 + j];

    /* Maintain current index of sub-arrays and main array */
    i = 0;
    j = 0;
    k = p;


    /* Until we reach either end of either L or M, pick larger among
    elements L and M and place them in the correct position at A[p..r] */
    while (i < n1 && j < n2) {
        if (L[i].value <= M[j].value) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = M[j];
            j++;
        }
        k++;
    }

    /* When we run out of elements in either L or M, pick up the remaining elements and put in A[p..r] */
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = M[j];
        j++;
        k++;
    }

    free(L);
    free(M);
}

/* Divide the array into two subarrays, sort them and merge them */
void mergeSort(eigen * arr, int l, int r) {
    if (l < r) {

        /* m is the point where the array is divided into two subarrays */
        int m = l + (r - l) / 2;

        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);

        /* Merge the sorted subarrays */
        merge(arr, l, m, r);
    }
}

int eigen_gap(eigen *eigen_items, int N) {
    int i, k;
    double lambda_i, max_diff;

    max_diff = -1;
    k = 0;

    for (i = 0; i < (N / 2); i++) {
        lambda_i = fabs(eigen_items[i].value - eigen_items[i + 1].value);
        if (lambda_i > max_diff) {
            max_diff = lambda_i;
            k = i;
        }
    }
    k += 1;
    return k;
}

double **gen_mat_k_eigenvectors(int N, int k, eigen *eigen_items) {
    int i, j;
    double **U;
    U = gen_mat(N, k);
    for (i = 0; i < k; i++) {
        double *eigen_vector = eigen_items[i].vector;
        for (j = 0; j < N; j++) {
            U[j][i] = eigen_vector[j];
        }
    }
    return U;
}

void normalize_mat(double **U, int N, int k) {
    int i, j;
    for (i = 0; i < N; i++) {
        double norm;
        norm = 0;
        for (j = 0; j < k; j++) {
            norm += pow(U[i][j], 2);
        }
        norm = pow(norm, 0.5);
        for (j = 0; j < k; j++) {
            U[i][j] = U[i][j] / norm;
        }
    }
}

int *get_N_dim_from_file(char *filename) {
    int i, N, dim, first_line;
    char *line;
    int *res;
    FILE *fp;
    size_t len;
    ssize_t read;

    N = 0;
    dim = 1;
    first_line = 1;
    line = NULL;
    len = 0;

    res = calloc(2, sizeof(int));
    assert_int_arr(res);

    fp = fopen(filename, "r");
    assert_fp(fp);
    /*
         calculating dim and N
    */
    while ((read = getline(&line, &len, fp)) != -1) {
        if (first_line) {
            for (i = 0; i < read; i++) {
                if (line[i] == ',') {
                    dim++;
                }
            }
            first_line = 0;
        }
        N++;
    }
    if (line) {
        free(line);
    }

    fclose(fp);
    res[0] = N;
    res[1] = dim;
    return res;
}

double **get_mat_from_file(char *filename, int N, int dim) {
    double n1;
    char c;
    int i, j;
    double **data_points;
    double *block;
    FILE *fp;

    j = 0;
    fp = fopen(filename, "r");
    assert_fp(fp);

    block = calloc(N * dim, sizeof(double));
    assert_double_arr(block);

    data_points = calloc(N, sizeof(double *));
    assert_double_mat(data_points);
    for (i = 0; i < N; i++) {
        data_points[i] = block + i * dim;
    }
    i = 0;
    while (fscanf(fp, "%lf%c", &n1, &c) == 2) {
        data_points[i][j] = n1;
        j++;
        if (c == '\n') {
            i++;
            j = 0;
        }
    }
    data_points[i][j] = n1;
    fclose(fp);
    return data_points;
}

void free_mat(double **mat) {
    free(mat[0]);
    free(mat);
}

/*PyObject *mat_to_Python_mat(double **mat, int N, int dim) {
    Py_ssize_t i, j, rows = N, columns = dim;
    PyObject *res = PyList_New(N);

    for (i = 0; i < rows; i++) {
        PyObject *item = PyList_New(dim);
        for (j = 0; j < columns; j++)
            PyList_SET_ITEM(item, j, PyFloat_FromDouble(mat[i][j]));
        PyList_SET_ITEM(res, i, item);
    }
    return res;
}*/
/*
PyObject * kmeans2_py(int k, int num_of_lines, int dim, PyObject *centroids_py,
                      PyObject *points_to_cluster_py, int centroids_length, int points_to_cluster_length) {
    return kmeans2(k, num_of_lines, dim, centroids_py,
                   points_to_cluster_py, centroids_length, points_to_cluster_length);
}*/

void V_multi_P(double ** V, double s, double c, int N, int i, int j) {
    /*since P is almost-diagonal, no need to perform full matrix multiplication*/
    int r;
    double V_ri, V_rj;

    for (r = 0; r < N; r++) {
        V_ri = V[r][i];
        V_rj = V[r][j];

        V[r][i] = (c * V_ri) - (s * V_rj);
        V[r][j] = (s * V_ri) + (c * V_rj);
    }
}

void assert_double_arr(const double * arr) {
    if (arr == NULL) {
        printf("An Error Has Occured");
        exit(0);
    }
}

void assert_int_arr(const int * arr) {
    if (arr == NULL) {
        printf("An Error Has Occured");
        exit(0);
    }
}

void assert_double_mat(double ** mat) {
    if (mat == NULL) {
        printf("An Error Has Occured");
        exit(0);
    }
}

void assert_eigen_arr(eigen * arr) {
    if (arr == NULL) {
        printf("An Error Has Occured");
        exit(0);
    }
}

void assert_fp(FILE * fp) {
    if (fp == NULL) {
        printf("An Error Has Occured");
        exit(0);
    }

}
