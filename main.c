#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>


// attempt to start spkmeans //

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
//        print_mat(D,N,N);
//        printf("\n");
        diag_mat_pow_half(D,N);
//        print_mat(D,N,N);
//        printf("\n");
        diag_mat_multi_reg_mat(D,W,N);
        reg_mat_multi_diag_mat(D,W,N);
        // need to add another D^0.5;
        identity_minus_reg_mat(W,N);
        print_mat(W,N,N);
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
    int i, j = 0;
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
    double * D_diag;
    D_diag = calloc(N,sizeof (double));
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

void identity_minus_reg_mat(double ** mat, int N) {
    int i,j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (i == j) {
                mat[i][j] = 1 - mat[i][j];
            }
            else {
                mat[i][j] *= -1;
            }
        }
    }
}

