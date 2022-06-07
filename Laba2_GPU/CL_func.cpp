#include "CL_func.h"
#include <iostream>
#include <string>

void saxpyCL(int n, float a, float* x, int incx, float* y, int incy, cl_device_id device) {
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    const char * src_saxpy =
        "__kernel void saxpy ( int n, float a,                 \n"\
        "   __global float *x, int incx,  __global float *y, int incy) {  \n"\
        "int i = get_global_id (0);                 \n"
        "if(i<n)                  \n"\
        "   y[i * incy] = y[i * incy] + a * x[i * incx];               \n"\
        "}\n";

    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, NULL);

    size_t srclen[] = { strlen(src_saxpy) };
    cl_program program = clCreateProgramWithSource(context, 1, &src_saxpy, srclen, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    //------------------------
    size_t len = 0;
    std::cout << "\nsucces build " << clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len) << std::endl;

    cl_kernel kernel = clCreateKernel(program, "saxpy", NULL);

    //input dATA

    cl_mem xBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n * incx, NULL, NULL);
    cl_mem yBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n * incy, NULL, NULL);

    clEnqueueWriteBuffer(queue, xBuf, CL_TRUE, 0, sizeof(float) * n * incx, x, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, yBuf, CL_TRUE, 0, sizeof(float) * n * incx, y, 0, NULL, NULL);

    //size_t count = { n };
    clSetKernelArg(kernel, 0, sizeof(int), &n);
    clSetKernelArg(kernel, 1, sizeof(float), &a);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &xBuf);
    clSetKernelArg(kernel, 3, sizeof(int), &incx);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &yBuf);
    clSetKernelArg(kernel, 5, sizeof(int), &incy);

    //size_t group = { 50 };
    size_t group;
    clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, NULL); //---------------------
    auto glob_work_size = static_cast<size_t>(n);
    size_t loc_work_size = { 128 };

    double begin = omp_get_wtime();
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &glob_work_size, &group, 0, NULL, NULL);
    clFinish(queue);
    double end = omp_get_wtime();
    std::cout << "GPU VERS float" << "\n" << (end - begin) << std::endl;
    


    clEnqueueReadBuffer(queue, yBuf, CL_TRUE, 0, sizeof(float) * n * incy, y, 0, NULL, NULL);

    clReleaseMemObject(xBuf);
    clReleaseMemObject(yBuf);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

void daxpyCL(int n, double a, double* x, int incx, double* y, int incy, cl_device_id device) {
    const char * src_daxpy =
        "__kernel void daxpy ( int n, double aD,                 \n"\
        "   __global double *x, int incx,  __global double *y, int incy) {  \n"\
        "int i = get_global_id (0);                 \n"
        "if(i<n)                  \n"\
        "   y[i * incy] = y[i * incy] + aD * x[i * incx];               \n"\
        "}\n"; 

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, NULL);

    size_t srclen[] = { strlen(src_daxpy) };
    cl_program program = clCreateProgramWithSource(context, 1, &src_daxpy, srclen, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    //------------------------
    size_t len = 0;
    std::cout << "\nsucces build " << clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len) << std::endl;

    cl_kernel kernel = clCreateKernel(program, "daxpy", NULL);

    //input dATA

    cl_mem xBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * n * incx, NULL, NULL);
    cl_mem yBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * n * incy, NULL, NULL);

    clEnqueueWriteBuffer(queue, xBuf, CL_TRUE, 0, sizeof(double) * n * incx, x, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, yBuf, CL_TRUE, 0, sizeof(double) * n * incx, y, 0, NULL, NULL);

    //size_t count = { n };
    clSetKernelArg(kernel, 0, sizeof(int), &n);
    clSetKernelArg(kernel, 1, sizeof(double), &a);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &xBuf);
    clSetKernelArg(kernel, 3, sizeof(int), &incx);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &yBuf);
    clSetKernelArg(kernel, 5, sizeof(int), &incy);

    //size_t group = { 50 };
    size_t group;
    clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, NULL);
    auto glob_work_size = static_cast<size_t>(n);
    size_t loc_work_size = { 128 };

    double begin = omp_get_wtime();
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &glob_work_size, &loc_work_size, 0, NULL, NULL);
    clFinish(queue);
    double end = omp_get_wtime();
    std::cout << "GPU VERS double" << "\n" << (end - begin) << std::endl;
    


    clEnqueueReadBuffer(queue, yBuf, CL_TRUE, 0, sizeof(double) * n * incy, y, 0, NULL, NULL);
    clReleaseMemObject(xBuf);
    clReleaseMemObject(yBuf);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}