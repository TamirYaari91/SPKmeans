#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#define eps 0.001

// maybe we can improve complexity of matrices functions if matrices are initialized to zero

double **wam(double **, int, int);

double **ddg(double **, int);

double norm(double *, double *, int);

void print_mat(double **, int, int);

double sum_row(const double *, int);

void diag_mat_pow_half(double **, int);

void diag_mat_multi_reg_mat(double **, double **, int); // result mat is W - the reg mat - the second mat!

void reg_mat_multi_diag_mat(double **, double **, int); // result mat is W - the reg mat - the second mat!

void identity_minus_reg_mat(double **, int);

void A_to_A_tag(double **, double **, int);

int *max_indices_off_diag(double **, int);

int sign(double);

double off(double **, int);

double **gen_id_mat(int);

double **jacobi(double **, int);

double **gen_P(double, double, int, int, int);

void multi_mat(double**, double**, int);

int main(int argc, char *argv[]) {
    double n1;
    char c;
    int i, k, N = 0, dim = 1, first_line = 1, j = 0;
    double **data_points;
    double *block;
    char *goal, *filename, *end_k, *line = NULL;
    FILE *fp;
    size_t len = 0;
    ssize_t read;


    if (argc < 3) {
        printf("Invalid Input!");
        return 0;
    }

    k = strtol(argv[1], &end_k, 10);
    if (*end_k != '\0' || k < 0) {
        printf("Invalid Input!");
        return 0;
    }
    printf("k = %d\n", k);
    goal = argv[2];
    filename = argv[3];
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
    if (line)
        free(line);

    // adding points to initial matrix //
    rewind(fp);
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

    printf("dim = %d\n", dim);
    printf("N = %d\n", N);
    if (k >= N) {
        printf("Invalid Input!");
        return 0;
    }

    // if k == 0 - need to implement heuristic 1.3 //


    // assuming k > 0 //

    if (strcmp(goal, "wam") == 0) {
        double **W = wam(data_points, N, dim);
        print_mat(W, N, N);
        return 0;
    }

    if (strcmp(goal, "ddg") == 0) {
        double **W = wam(data_points, N, dim);
        double **D = ddg(W, N);
        print_mat(D, N, N);
        return 0;
    }

    if (strcmp(goal, "lnorm") == 0) {
        double **W = wam(data_points, N, dim);
        double **D = ddg(W, N);
        diag_mat_pow_half(D, N);
        diag_mat_multi_reg_mat(D, W, N);
        reg_mat_multi_diag_mat(D, W, N);
        identity_minus_reg_mat(W, N);
        print_mat(W, N, N);
        return 0;
    }

    if (strcmp(goal, "jacobi") == 0) {
        double **W = wam(data_points, N, dim);
        double **D = ddg(W, N);
        diag_mat_pow_half(D, N);
        diag_mat_multi_reg_mat(D, W, N);
        reg_mat_multi_diag_mat(D, W, N);
        identity_minus_reg_mat(W, N);
        printf("lnorm = \n");
        W[0][0] = 3;
        W[0][1] = 2;
        W[0][2] = 4;
        W[1][0] = 2;
        W[1][1] = 0;
        W[1][2] = 2;
        W[2][0] = 4;
        W[2][1] = 2;
        W[2][2] = 3;
        print_mat(W, N, N);
        printf("\n");
        jacobi(W,N);
        return 0;
    }

    // need to add big "else" in case goal is not any of the valid options

    return 0;
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
            printf("%lf", mat[row][columns]);
            if (columns == dim - 1) {
                printf("\n");
            } else {
                printf(",");
            }

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
    theta = (A[j][j] - A[i][i]) / 2 * A[i][j];
    t = sign(theta) / (fabs(theta) + sqrt((pow(theta, 2)) + 1));
    c = 1 / sqrt((pow(t, 2)) + 1);
    s = t * c;
    double **P = gen_P(s,c,i,j,N);
    multi_mat(V,P,N);
    printf("V = \n");
    print_mat(V,N,N);
    printf("\n");


    printf("t = %lf\n", t);
    printf("s = %lf\n", s);
    printf("c = %lf\n", c);

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
    printf("A' = \n");
    print_mat(A, N, N);
    printf("\n");

}

int *max_indices_off_diag(double **A, int N) {
    double res = 0;
    int i, j, max_i = 0, max_j = 0;
    int *arr = calloc(2, sizeof(double));
    assert(arr);

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (i != j) {
                if (fabs(A[i][j]) > res) {
                    res = fabs(A[i][j]);
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

double **jacobi(double **A, int N) {
    int i, j, iter = 0, max_iter = 100;
    double **V = gen_id_mat(N);
    double diff = MAXFLOAT;
        while (diff > eps && iter < max_iter) {
        iter++;
        double off_A = off(A, N);
        A_to_A_tag(A, V, N);
        diff = off_A - off(A, N);

    }


}

double **gen_P(double s, double c, int i, int j, int N) {
    double **P = gen_id_mat(N);
    P[i][j] = s;
    P[j][i] = -s;
    P[i][i] = c;
    P[j][j] = c;
    return P;
}

void multi_mat(double** mat1, double** mat2, int N) {
        int i, j, k;
        double ** res = gen_id_mat(N);
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
