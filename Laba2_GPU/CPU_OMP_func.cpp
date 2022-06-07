#include "CPU_OMP_func.h"

void saxpy_CPU(int n, float a, float* x, int incx, float* y, int incy) {
    for (int i = 0; i < n; i++) {
        y[i * incy] = y[i * incy] + a * x[i * incx];
    }
}

void daxpy_CPU(int n, double a, double* x, int incx, double* y, int incy) {
    for (int i = 0; i < n; i++) {
        y[i * incy] = y[i * incy] + a * x[i * incx];
    }
}

void saxpy_OMP(int n, float a, float* x, int incx, float* y, int incy) {
    #pragma omp parallel 
    {
        #pragma omp for
        for (int i = 0; i < n; i++) {
            y[i * incy] = y[i * incy] + a * x[i * incx];
        }
    }
}

void daxpy_OMP(int n, double a, double* x, int incx, double* y, int incy) {
    #pragma omp parallel 
    {
        #pragma omp for
        for (int i = 0; i < n; i++) {
            y[i * incy] = y[i * incy] + a * x[i * incx]; 
        }
    }
}
