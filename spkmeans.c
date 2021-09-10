#define _GNU_SOURCE

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

#define eps pow(10,-15)


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

    spkmeans_C(argv[3], argv[2], k);
    return 0;
}

void spkmeans_C(char *filename, char *goal, int k) {
    /*Main algorithms driver method - activates functions according to the goal, k and filename as entered*/
    int N, dim;
    int *N_dim;

    N_dim = get_N_dim_from_file(filename);
    N = N_dim[0];
    dim = N_dim[1];

    if (k >= N) {
        free(N_dim);
        printf("Invalid Input!");
        return;
    }

    if (strcmp(goal, "wam") == 0) {
        wam_wrapper(filename, N, dim);
    } else if (strcmp(goal, "ddg") == 0) {
        ddg_wrapper(filename, N, dim);
    } else if (strcmp(goal, "lnorm") == 0) {
        lnorm_wrapper(filename, N, dim);
    } else if (strcmp(goal, "jacobi") == 0) {
        jacobi_wrapper(filename, N);
    } else if (strcmp(goal, "spk") == 0) {
        spk_wrapper(filename, N, dim, k);

    } else { /* goal is not any of the valid options */
        printf("Invalid Input!");
    }
    free(N_dim);
}

void wam_wrapper(char *filename, int N, int dim) {
    /*wrapper for goal == wam - gets data points from file and calculates wam according to given logic*/
    double **data_points, **W;

    data_points = get_mat_from_file(filename, N, dim);
    W = wam(data_points, N, dim);
    print_mat(W, N, N);
    free_mat(W);
    free_mat(data_points);
}

void ddg_wrapper(char *filename, int N, int dim) {
    /*wrapper for goal == ddg - gets data points from file and calculates wam according to given logic*/
    double **data_points, **W, **D;

    data_points = get_mat_from_file(filename, N, dim);
    W = wam(data_points, N, dim);
    D = ddg(W, N);
    print_mat(D, N, N);
    free_mat(W);
    free_mat(D);
    free_mat(data_points);
}

void lnorm_wrapper(char *filename, int N, int dim) {
    /*wrapper for goal == lnorm - gets data points from file and calculates wam according to given logic*/
    double **data_points, **W, **D;

    data_points = get_mat_from_file(filename, N, dim);
    W = wam(data_points, N, dim);
    D = ddg(W, N);
    lnorm(W, D, N);
    print_mat(W, N, N);
    free_mat(W);
    free_mat(D);
    free_mat(data_points);
}

void jacobi_wrapper(char *filename, int N) {
    /*wrapper for goal == jacobi - gets data points from file and calculates wam according to given logic*/
    double **W, **V;
    double *eigenvalues;
    int i;

    W = get_mat_from_file(filename, N, N);
    V = jacobi(W, N);
    eigenvalues = get_diag(W, N);
    print_row(eigenvalues, N);
    printf("\n");
    for (i = 0; i < N; i++) {
        double *eigenvector = get_ith_column(V, i, N);
        print_row(eigenvector, N);
        if (i < N - 1) {
            printf("\n");
        }
        free(eigenvector);
    }
    free_mat(W);
    free_mat(V);
    free(eigenvalues);
}

void spk_wrapper(char *filename, int N, int dim, int k) {
    /*wrapper for goal == spk - gets data points from file and calculates wam according to given logic*/
    double **data_points, **W, **V, **D, **U;
    double *eigenvalues;
    int i;
    eigen *eigen_items; /*struct containing eigen value paired to its respective eigen vector*/

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
    mergeSort(eigen_items, 0, N - 1); /*sorting the eigen_items according to eigen values - using mergeSort since it
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
    kmeans(U, k, N); /*KMeans from EX1*/
    free_mat(U);
}


double norm(double *p1, double *p2, int dim) {
    /*calculates Euclidean distance between p1 and p2 - both points in R^dim*/
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
    /*Calculates and prints the Weighted Adjacency Matrix*/
    int i, j;
    double **W;

    j = 0;
    W = gen_mat(N, N);
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

void print_mat(double **mat, int N, int dim) {
    /*Prints matrix according to given dimensions: N == number of rows, dim == number of columns*/
    int row, columns;
    for (row = 0; row < N; row++) {
        for (columns = 0; columns < dim; columns++) {
            print_double(mat[row][columns]);
            if (columns == dim - 1) {
                if (row < N - 1) {
                    printf("\n");
                }
            } else {
                printf(",");
            }
        }
    }
}

void print_double(double num) {
    /*Prevents "-0.0000" situation*/
    if ((num < 0) && (num > -0.00005)) {
        printf("0.0000");
    } else {
        printf("%.4f", num);
    }
}

void print_row(double *row, int len) {
    /*Prints array (row) according to given dimension: len == number of items*/
    int i;
    for (i = 0; i < len; i++) {
        print_double(row[i]);
        if (i != len - 1) {
            printf(",");
        }
    }
}

double **ddg(double **wam_mat, int N) {
    /*Calculates and outputs the Diagonal Degree Matrix*/
    int i, j;
    double **D;

    D = gen_mat(N, N);
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (i == j) {
                D[i][j] = sum_row(wam_mat[i], N);
            }
        }
    }
    return D;
}

void lnorm(double **W, double **D, int N) {
    /*Performs W = I -D^0.5*W*D^0.5*/
    diag_mat_pow_half(D, N); /*D = D^0.5*/
    diag_mat_multi_reg_mat(D, W, N); /*W = DW*/
    reg_mat_multi_diag_mat(D, W, N); /*W = WD*/
    identity_minus_reg_mat(W, N); /*W = I -W*/
}

double sum_row(const double *mat, int m) {
    /*Sums up values in double array (row) of size m*/
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

void diag_mat_multi_reg_mat(double **D, double **W, int N) {
    /*W = DW*/
    /*result mat is W - the reg mat - the second mat!*/
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
    /*W = WD*/
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
    /*W = I-W*/
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
    /*Calculates A' from A using the relation between them as explained in 1.2.6*/
    int i, j, r;
    int *arr_max;
    double theta, s, t, c, a_ri, a_rj, a_ii, a_jj;

    arr_max = max_indices_off_diag(A, N); /*Finding pivot - 1.2.3*/
    i = arr_max[0];
    j = arr_max[1];
    theta = (A[j][j] - A[i][i]) / (2 * A[i][j]);
    t = sign(theta) / (fabs(theta) + sqrt((pow(theta, 2)) + 1));
    c = 1 / sqrt((pow(t, 2)) + 1);
    s = t * c;
    V_multi_P(V, s, c, N, i, j); /*V = VP*/

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
    /*1.2.3*/
    a_ii = pow(c, 2) * A[i][i] + pow(s, 2) * A[j][j] - 2 * s * c * A[i][j];
    a_jj = pow(c, 2) * A[j][j] + pow(s, 2) * A[i][i] + 2 * s * c * A[i][j];
    A[j][j] = a_jj;
    A[i][i] = a_ii;
    A[i][j] = 0.0;
    A[j][i] = 0.0;

    free(arr_max);
}

int *max_indices_off_diag(double **A, int N) {
    /*returns [i,j] so that A[i][j] is the off-diagonal largest absolute element in A*/
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
    /*Calculates Off(A)^2*/
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
    /*Generates ID matrix*/
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
        }
    }
    return I;
}

double **gen_mat(int N, int k) {
    /*Generates Nxk matrix of zeroes*/
    int i;
    double **M;
    double *block;

    block = calloc(N * k, sizeof(double));
    assert_double_arr(block);
    M = calloc(N, sizeof(double *));
    assert_double_mat(M);
    for (i = 0; i < N; i++) {
        M[i] = block + i * k;
    }
    return M;
}


double **jacobi(double **A, int N) {
    /*Calculates and prints the eigenvalues and eigenvectors as described in 1.2.1*/
    int iter, max_iter;
    double diff;
    double **V;

    iter = 0;
    max_iter = 100;
    V = gen_id_mat(N);
    diff = 1;

    while (diff > eps && iter < max_iter) { /*Stops when off(A)^2- - off(A')^2 <= epsilon OR 100 iterations*/
        double off_A = off(A, N);
        iter++;
        A_to_A_tag(A, V, N); /*Also change V during A->A' iteration so that eventually V = P1*P2*P3*...*/
        diff = off_A - off(A, N);
    }
    return V;
}

double *get_diag(double **mat, int N) {
    /*returns [mat[0][0],mat[1][1],...,mat[N-1][N-1]]*/
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
    /*returns [mat[0][col_ind],mat[1][col_ind],...,mat[N-1][col_ind]]*/
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
void merge(eigen *arr, int p, int q, int r) {
    int i, j, k, n1, n2;
    eigen *L, *M;


    /* Create L ← A[p..q] and M ← A[q+1..r] */
    n1 = q - p + 1;
    n2 = r - q;

    L = calloc(n1, sizeof(eigen));
    assert_eigen_arr(L);
    M = calloc(n2, sizeof(eigen));
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
void mergeSort(eigen *arr, int l, int r) {
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
    /*Calculates k using Eigangap Heuristic*/
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
    k += 1; /*Adjust index*/
    return k;
}

double **gen_mat_k_eigenvectors(int N, int k, eigen *eigen_items) {
    /*Creates Nxk using the first k eigenvectors as columns - as described in step 4 of the algorithm*/
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
    /*Normalizes Nxk matrix  - as described in step 5 of the algorithm*/
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
    /*Goes over the file once to retrieve N == number of rows and dim == number of columns in file matrix*/
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

    /*calculating dim and N*/
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
    /*Using calculated N,dim goes over file to retrieve the matrix itself*/
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
    /*Since the matrices are allocated as a contigous block, we first free the block of N*k size and then the pointer*/
    free(mat[0]);
    free(mat);
}

void V_multi_P(double **V, double s, double c, int N, int i, int j) {
    /*V = VP; since P is almost-diagonal, no need to perform full matrix multiplication*/
    int r;
    double V_ri, V_rj;

    for (r = 0; r < N; r++) {
        V_ri = V[r][i];
        V_rj = V[r][j];

        V[r][i] = (c * V_ri) - (s * V_rj);
        V[r][j] = (s * V_ri) + (c * V_rj);
    }
}

void assert_double_arr(const double *arr) {
    /*Replaces assert*/
    if (arr == NULL) {
        printf("An Error Has Occured");
        exit(0);
    }
}

void assert_int_arr(const int *arr) {
    /*Replaces assert*/
    if (arr == NULL) {
        printf("An Error Has Occured");
        exit(0);
    }
}

void assert_double_mat(double **mat) {
    /*Replaces assert*/
    if (mat == NULL) {
        printf("An Error Has Occured");
        exit(0);
    }
}

void assert_eigen_arr(eigen *arr) {
    /*Replaces assert*/
    if (arr == NULL) {
        printf("An Error Has Occured");
        exit(0);
    }
}

void assert_fp(FILE *fp) {
    /*Replaces assert*/
    if (fp == NULL) {
        printf("An Error Has Occured");
        exit(0);
    }
}
