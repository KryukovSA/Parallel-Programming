#pragma once
#include <omp.h>

void saxpy_CPU(int n, float a, float* x, int incx, float* y, int incy);
void daxpy_CPU(int n, double a, double* x, int incx, double* y, int incy);
void saxpy_OMP(int n, float a, float* x, int incx, float* y, int incy);
void daxpy_OMP(int n, double a, double* x, int incx, double* y, int incy);