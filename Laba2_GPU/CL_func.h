#pragma once
#include <CL/cl.h>
#include <omp.h>

void saxpyCL(int n, float a, float* x, int incx, float* y, int incy, cl_device_id device);
void daxpyCL(int n, double a, double* x, int incx, double* y, int incy, cl_device_id device);