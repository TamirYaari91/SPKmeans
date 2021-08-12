#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include "spkmeans.h"
#include "kmeans.h"

#define eps 0.001

// maybe we can improve complexity of matrices functions if matrices are initialized to zero

// need to free memory


int main(int argc, char *argv[]) {
    int i, k, N, dim;
    char *goal, *filename, *end_k;
    double **data_points;

    if (argc < 3) {
        printf("Invalid Input!");
        return 0;
    }

    k = (int) strtol(argv[1], &end_k, 10);
    if (*end_k != '\0' || k < 0) {
        printf("Invalid Input!");
        return 0;
    }
    goal = argv[2];
    filename = argv[3];

    int *N_dim = get_N_dim_from_file(filename);

    N = N_dim[0];
    dim = N_dim[1];

    if (k >= N) {
        printf("Invalid Input!");
        return 0;
    }

    if (strcmp(goal, "wam") == 0) {
        data_points = get_mat_from_file(filename, N, dim);
        double **W = wam(data_points, N, dim);
        print_mat(W, N, N);
        return 0;
    }

    if (strcmp(goal, "ddg") == 0) {
        data_points = get_mat_from_file(filename, N, dim);
        double **W = wam(data_points, N, dim);
        double **D = ddg(W, N);
        print_mat(D, N, N);
        return 0;
    }

    if (strcmp(goal, "lnorm") == 0) {
        data_points = get_mat_from_file(filename, N, dim);
        double **W = wam(data_points, N, dim);
        double **D = ddg(W, N);
        lnorm(W, D, N);
        print_mat(W, N, N);
        return 0;
    }

    if (strcmp(goal, "jacobi") == 0) {
        double **W = get_mat_from_file(filename,N,N);
        double **V;
        V = jacobi(W, N);
        double *eigenvalues = get_diag(W, N);
        print_row(eigenvalues, N);
        printf("\n");
        eigen *eigen_items = calloc(N, sizeof(eigen));
        assert(eigen_items);
        for (i = 0; i < N; i++) {
            double *eigenvector = get_ith_column(V, i, N);
            print_row(eigenvector, N);
            if (i != N - 1) {
                printf("\n");
            }
            eigen item;
            item.value = eigenvalues[i];
            item.vector = eigenvector;
            eigen_items[i] = item;
        }
        return 0;
    }
    if (strcmp(goal, "spk") == 0) {
        data_points = get_mat_from_file(filename, N, dim);
        double **W = wam(data_points, N, dim);
        double **D = ddg(W, N);
        lnorm(W, D, N);
        double **V = jacobi(W, N);
        double *eigenvalues = get_diag(W, N);
        eigen *eigen_items = calloc(N, sizeof(eigen));
        assert(eigen_items);
        for (i = 0; i < N; i++) {
            double *eigenvector = get_ith_column(V, i, N);
            eigen item;
            item.value = eigenvalues[i];
            item.vector = eigenvector;
            eigen_items[i] = item;
        }
        if (k == 0) {
            k = eigen_gap(eigen_items, N);
        }
        double **U;
        U = mat_k_eigenvectors(N, k, eigen_items);
        normalize_mat(U, N, k);
        kmeans(U, k, N);
        return 0;
    }

    else { // goal is not any of the valid options
        printf("Invalid Input!");
        return 0;
    }
}

double norm(double *p1, double *p2, int dim) {
    int i;
    double res, sum = 0;
    for (i = 0; i < dim; i++) {
        sum += pow(p1[i] - p2[i], 2);
    }
    res = sqrt(sum);
    return res;
}

double **wam(double **data_points, int N, int dim) {
    int i, j = 0;
    double **W;
    double *block;

    block = calloc(N * N, sizeof(double));
    assert(block);
    W = calloc(N, sizeof(double *));
    assert(W);
    for (i = 0; i < N; i++) {
        W[i] = block + i * N;
        W[i][i] = 0.0; // W_ii = 0
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

void print_mat(double **mat, int N, int dim) {
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
}

void print_row(double *row, int len) {
    int i;
    for (i = 0; i < len; i++) {
        printf("%.4lf", row[i]);
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
    assert(block);
    D = calloc(N, sizeof(double *));
    assert(D);
    for (i = 0; i < N; i++) {
        D[i] = block + i * N;
        for (j = 0; j < N; j++) {
            if (i == j) {
                D[i][j] = sum_row(wam_mat[i], N);
            } else {
                D[i][j] = 0.0;
            }
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
    double res = 0;
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

void diag_mat_multi_reg_mat(double **D, double **W, int N) { // result mat is W - the reg mat - the second mat!
    int i, j;
    double d_ii;

    for (i = 0; i < N; i++) {
        d_ii = D[i][i];
        for (j = 0; j < N; j++) {
            W[i][j] *= d_ii;
        }
    }
}

void reg_mat_multi_diag_mat(double **D, double **W, int N) { // result mat is W - the reg mat - the second mat!
    int i, j;
    double *D_diag;
    D_diag = calloc(N, sizeof(double));
    assert(D_diag);
    for (i = 0; i < N; i++) {
        D_diag[i] = D[i][i];
    }
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            W[i][j] *= D_diag[j];
        }
    }
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

    arr_max = max_indices_off_diag(A, N);
    i = arr_max[0];
    j = arr_max[1];
    theta = (A[j][j] - A[i][i]) / (2 * A[i][j]);
    t = sign(theta) / (fabs(theta) + sqrt((pow(theta, 2)) + 1));
    c = 1 / sqrt((pow(t, 2)) + 1);
    s = t * c;
    double **P = gen_P(s, c, i, j, N);
    multi_mat(V, P, N);

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
    double val = -1;
    int i, j, max_i = 0, max_j = 0;
    int *arr = calloc(2, sizeof(double));
    assert(arr);


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
    double res = 0;
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
    assert(block);
    I = calloc(N, sizeof(double *));
    assert(I);
    for (i = 0; i < N; i++) {
        I[i] = block + i * N;
        for (j = 0; j < N; j++) {
            if (i == j) {
                I[i][j] = 1.0;
            } else {
                I[i][j] = 0.0;
            }
        }
    }
    return I;
}

double **gen_mat(int N, int k) {
    int i, j;
    double **M;
    double *block;

    block = calloc(N * k, sizeof(double));
    assert(block);
    M = calloc(N, sizeof(double *));
    assert(M);
    for (i = 0; i < N; i++) {
        M[i] = block + i * k;
        //maybe not necessary:
        for (j = 0; j < k; j++) {
            M[i][j] = 0.0;
        }
    }
    return M;
}


double **jacobi(double **A, int N) {
    int iter = 0, max_iter = 100;
    double **V = gen_id_mat(N);
    double diff = MAXFLOAT;
    while (diff > eps && iter < max_iter) {
        iter++;
        double off_A = off(A, N);
        A_to_A_tag(A, V, N);
        diff = off_A - off(A, N);
    }
    return V;
}

double **gen_P(double s, double c, int i, int j, int N) {
    double **P = gen_id_mat(N);
    P[i][j] = s;
    P[j][i] = -s;
    P[i][i] = c;
    P[j][j] = c;
    return P;
}

void multi_mat(double **mat1, double **mat2, int N) {
    int i, j, k;
    double **res = gen_id_mat(N);
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
}

double *get_diag(double **mat, int N) {
    int i;
    double *diag;
    diag = calloc(N, sizeof(double));
    assert(diag);

    for (i = 0; i < N; i++) {
        diag[i] = mat[i][i];
    }
    return diag;
}

double *get_ith_column(double **mat, int col_ind, int N) {
    int i;
    double *col = calloc(N, sizeof(double));
    assert(col);
    for (i = 0; i < N; i++) {
        col[i] = mat[i][col_ind];
    }
    return col;
}

// A function to implement bubble sort, modified to sort eigen_items - GeeksForGeeks.org

void swap(eigen *xp, eigen *yp) {
    eigen temp = *xp;
    *xp = *yp;
    *yp = temp;
}

void eigen_bubble_sort(eigen *arr, int n) {
    int i, j;
    int swapped;
    for (i = 0; i < n - 1; i++) {
        swapped = 0;
        for (j = 0; j < n - i - 1; j++) {
            if (arr[j].value > arr[j + 1].value) {
                swap(&arr[j], &arr[j + 1]);
                swapped = 1;
            }
        }
        if (swapped == 0) {
            break;
        }
    }
}

int eigen_gap(eigen *eigen_items, int N) {
    int i, k = 0;
    double lambda_i, max_diff = -1;
    eigen_bubble_sort(eigen_items, N);
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

double **mat_k_eigenvectors(int N, int k, eigen *eigen_items) {
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
    for (i = 0; i < k; i++) {
        double norm = 0;
        for (j = 0; j < N; j++) {
            norm += pow(U[j][i], 2);
        }
        norm = pow(norm, 0.5);
        for (j = 0; j < N; j++) {
            U[j][i] = U[j][i] / norm;
        }
    }
}

int *get_N_dim_from_file(char *filename) {
    int i, N = 0, dim = 1, first_line = 1;
    char *line = NULL;
    FILE *fp;
    size_t len = 0;
    ssize_t read;

    int *res = calloc(2, sizeof(int));
    assert(res);

    fp = fopen(filename, "r");
    assert(fp != NULL);

    // calculating dim and N //
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
    int i, j = 0;
    double **data_points;
    double *block;
    FILE *fp;

    fp = fopen(filename, "r");
    assert(fp != NULL);

    block = calloc(N * dim, sizeof(double));
    assert(block);
    data_points = calloc(N, sizeof(double *));
    assert(data_points);
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


