#include <iostream>
#include "CL_func.h"
#include "CPU_OMP_func.h"
#include <vector>


using namespace std;

int main() {
    const int n = 512 * 100000;
    const int incx = 1;
    const int incy = 1;
    const size_t size_x = n * incx;
    const size_t size_y = n * incy;
    const float a = 0.01;
    const double aD = 0.01;

    vector<float> xFloat(size_x);
    vector<float> yFloat(size_y);
    vector<double> xDouble(size_x);
    vector<double> yDouble(size_y);
    vector<double> for_check_correct;

    for (int i = 0; i < size_x; i++) {
        xFloat[i] = (float)i;
        xDouble[i] = (double)i;
    }

    for (int i = 0; i < size_y; i++) {
        yFloat[i] = (float)i;
        yDouble[i] = (double)i;
    }









    //OMP vers-----------------------------------
    {
        auto yF = yFloat;
        double begin = omp_get_wtime();
        saxpy_OMP(n, a, xFloat.data(), incx, yF.data(), incy);
        double end = omp_get_wtime();
        cout << "OMP VERS float" << "\n" << (end - begin) << endl;
    }
    { 
        auto yD = yDouble;
        double begin = omp_get_wtime();
        daxpy_OMP(n, aD, xDouble.data(), incx, yD.data(), incy);
        double end = omp_get_wtime();
        cout << "OMP VERS double" << "\n" << (end - begin) << endl;
        for_check_correct = yD;
    }


    //sequential vers----------------------------
    {
        auto yF = yFloat;
        double begin = omp_get_wtime();
        saxpy_CPU(n, a, xFloat.data(), incx, yF.data(), incy);
        double end = omp_get_wtime();
        cout << "SEQ VERS float" << "\n" << (end - begin) << endl;
    }
    {
        auto yD = yDouble;
        double begin = omp_get_wtime();
        daxpy_CPU(n, aD, xDouble.data(), incx, yD.data(), incy);
        double end = omp_get_wtime();
        cout << "SEQ VERS double" << "\n" << (end - begin) << endl;
        if (yD == for_check_correct)
            cout << "correct result" << endl;
    }


    //CL vers--------------------------

 /*   const char * src_saxpy =
        "__kernel void saxpy ( int n, float a,                 \n"\
        "   __global float *x, int incx,  __global float *y, int incy) {  \n"\
        "int i = get_global_id (0);                 \n"
        "if(i<n)                  \n"\
        "   y[i * incy] = y[i * incy] + a * x[i * incx];               \n"\
        "}\n";*/

   /* const char * src_daxpy =
        "__kernel void daxpy ( int n, double aD,                 \n"\
        "   __global double *x, int incx,  __global double *y, int incy) {  \n"\
        "int i = get_global_id (0);                 \n"
        "if(i<n)                  \n"\
        "   y[i * incy] = y[i * incy] + aD * x[i * incx];               \n"\
        "}\n";/*/


    cl_uint platformCount = 0;
    clGetPlatformIDs(0, nullptr, &platformCount);
    cl_platform_id platform1 = NULL;
    if (0 < platformCount) {
        cl_platform_id* platform = new cl_platform_id[platformCount];
        clGetPlatformIDs(platformCount, platform, nullptr);

        platform1 = platform[0];
        delete[] platform;
    }
    //////////////////////////

    cl_platform_id* platform2 = new cl_platform_id[platformCount];
    clGetPlatformIDs(platformCount, platform2, nullptr);
    for (cl_uint i = 0; i < platformCount; ++i) {
        char platformName[128];
        clGetPlatformInfo(platform2[i], CL_PLATFORM_NAME,
            128, platformName, nullptr);
        std::cout << platformName << std::endl;
    }

    cl_uint num_all_devices = 0;

    cl_device_id device = NULL;
    char deviceName[128];
    clGetDeviceIDs(platform1, CL_DEVICE_TYPE_GPU, 1, &device, &num_all_devices);
    clGetDeviceInfo(device, CL_DEVICE_NAME, 128, deviceName, NULL);
    
    /*cl_context_properties properties[3] = {
    CL_CONTEXT_PLATFORM, (cl_context_properties)platform1, 0 };

    cl_context context = clCreateContextFromType((NULL == platform1) ? NULL : properties,
        CL_DEVICE_TYPE_GPU,
        NULL,
        NULL,
        NULL
    );

    size_t size;
    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);
    //----------------------------------------
    cl_device_id device = NULL;
    if (size > 0) {
        cl_device_id * devices = (cl_device_id *)alloca(size);
        clGetContextInfo(context, CL_CONTEXT_DEVICES, size, devices, NULL);
        device = devices[0]; //2 devices
    //}
        cout << "devices \n";
        for (cl_uint i = 0; i < 2; ++i) {
            char deviceName[128];
            clGetDeviceInfo(devices[i], CL_DEVICE_NAME,
                128, deviceName, nullptr);
            cout << deviceName << endl;
        }
    }*/

    
    /*
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, NULL);

    size_t srclen[] = { strlen(src_saxpy) };
    cl_program program = clCreateProgramWithSource(context, 1, &src_saxpy, srclen, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    //------------------------
    size_t len = 0;
    cout << "\nsucces build " << clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len) << endl;

    cl_kernel kernel = clCreateKernel(program, "saxpy", NULL);

    //input dATA

    cl_mem xBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n * incx, NULL, NULL);
    cl_mem yBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n * incy, NULL, NULL);

    clEnqueueWriteBuffer(queue, xBuf, CL_TRUE, 0, sizeof(float) * n * incx, xFloat.data(), 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, yBuf, CL_TRUE, 0, sizeof(float) * n * incx, yFloat.data(), 0, NULL, NULL);

    size_t count = { n };
    clSetKernelArg(kernel, 0, sizeof(int), &count);
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

    begin = omp_get_wtime();
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &glob_work_size, &group, 0, NULL, NULL);
    end = omp_get_wtime();
    cout << "GPU VERS float" << "\n" << (end - begin) << endl;
    clFinish(queue);


    clEnqueueReadBuffer(queue, yBuf, CL_TRUE, 0, sizeof(float) * n * incy, yFloat.data(), 0, NULL, NULL);*/
    
    

    //------------------------------------------------------------------------------------------------------
    /*
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, NULL);

    size_t srclen[] = { strlen(src_daxpy) };
    cl_program program = clCreateProgramWithSource(context, 1, &src_daxpy, srclen, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    //------------------------
    size_t len = 0;
    cout << "\nsucces build " << clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len) << endl;

    cl_kernel kernel = clCreateKernel(program, "daxpy", NULL);

    //input dATA

    cl_mem xBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * n * incx, NULL, NULL);
    cl_mem yBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * n * incy, NULL, NULL);

    clEnqueueWriteBuffer(queue, xBuf, CL_TRUE, 0, sizeof(double) * n * incx, xDouble.data(), 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, yBuf, CL_TRUE, 0, sizeof(double) * n * incx, yDouble.data(), 0, NULL, NULL);

    size_t count = { n };
    clSetKernelArg(kernel, 0, sizeof(int), &count);
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

    begin = omp_get_wtime();
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &glob_work_size, &loc_work_size, 0, NULL, NULL);
    end = omp_get_wtime();
    cout << "GPU VERS double" << "\n" << (end - begin) << endl;
    clFinish(queue);


    clEnqueueReadBuffer(queue, yBuf, CL_TRUE, 0, sizeof(double) * n * incy, yDouble.data(), 0, NULL, NULL);
    */


    {
        auto yF = yFloat;
        saxpyCL(n, a, xFloat.data(), incx, yF.data(), incy, device);
        
    }
    
    {
        auto yD = yDouble;
        daxpyCL(n, aD, xDouble.data(), incx, yD.data(), incy, device);
        if (yD == for_check_correct)
            cout << "correct result" << endl;

    }


   
    return 0;
}

