#ifndef UNTITLED10_SPKMEANS_H
#define UNTITLED10_SPKMEANS_H

/*
#include <Python/Python.h>
*/
#include <python3.7/Python.h>


typedef struct {
    double value;
    double *vector;
} eigen;

int kmeans(double **, int, int);

PyObject * spkmeans(char* , char*, int, int);

double **wam(double **, int, int);

double **ddg(double **, int);

void lnorm(double **, double **, int N);

double norm(double *, double *, int);

void print_mat(double **, int, int);

void print_row(double *, int);

double sum_row(const double *, int);

void diag_mat_pow_half(double **, int);

void diag_mat_multi_reg_mat(double **, double **, int); /*result mat is W - the reg mat - the second mat!*/

void reg_mat_multi_diag_mat(double **, double **, int); /*result mat is W - the reg mat - the second mat!*/

void identity_minus_reg_mat(double **, int);

void A_to_A_tag(double **, double **, int);

int *max_indices_off_diag(double **, int);

int sign(double);

double off(double **, int);

double **gen_id_mat(int);

double **jacobi(double **, int);

double **gen_P(double, double, int, int, int);

void multi_mat(double **, double **, int);

void swap(eigen *, eigen *);

void eigen_bubble_sort(eigen *, int);

double *get_diag(double **, int);

double *get_ith_column(double **, int, int);

int eigen_gap(eigen *, int);

double **gen_mat(int, int);

double **gen_mat_k_eigenvectors(int, int, eigen *);

void normalize_mat(double **, int, int);

int *get_N_dim_from_file(char *);

double ** get_mat_from_file(char *, int, int);

void free_mat (double **);

PyObject *mat_to_Python_mat(double **mat, int, int);

#endif /*UNTITLED10_SPKMEANS_H*/
