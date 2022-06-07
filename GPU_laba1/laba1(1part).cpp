#define _CRT_SECURE_NO_WARNINGS
#include <CL/cl.h>
#include <iostream>
#include <fstream>

using namespace std;

int main() {
 

    const char * src =
        " __kernel void printText() {\n"
        "int N = get_group_id(0);\n"
        "int M = get_local_id(0);\n"
        "int K = get_global_id(0);\n"
        "printf(\"I am from %d block, %d thread(global index : %d)\", N, M, K); \n"
        "}\n";
    
         
    cl_uint platformCount = 0;
    clGetPlatformIDs(0, nullptr, &platformCount);
    cl_platform_id platform1 = NULL;
    if (0 < platformCount) {
        cl_platform_id* platform = new cl_platform_id[platformCount];
        clGetPlatformIDs(platformCount, platform, nullptr);

        platform1 = platform[0];
        delete[] platform;
    }

    //--------------------------- cl build program использоватьс стоит
    cl_context_properties properties[3] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform1, 0 };

    cl_context context = clCreateContextFromType((NULL == platform1) ? NULL : properties,
        CL_DEVICE_TYPE_GPU,
        NULL,
        NULL,
        NULL
    );

    size_t size = 0;
    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);
    //----------------------------------------
    cl_device_id device = NULL;
    if (size > 0) {
        cl_device_id * devices = (cl_device_id *)alloca(size);
        clGetContextInfo(context, CL_CONTEXT_DEVICES, size, devices, NULL);
        device = devices[0];
    }
    
    //сл50
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, NULL);

    size_t srclen[] = { strlen(src) };


   
    cl_program program  = clCreateProgramWithSource(context, 1, &src, srclen, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    //------------------------
    size_t len = 0;
    cout << "succes build " << clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len) << endl;
    //--------------------------------
    cl_kernel kernel = clCreateKernel(program, "printText", NULL);
    
    //input dATA
  
    cl_mem input = clCreateBuffer(context, CL_MEM_READ_WRITE, NULL, NULL, NULL);//доступ см
    cl_mem output = clCreateBuffer(context, CL_MEM_READ_WRITE, NULL, NULL, NULL);

    clEnqueueWriteBuffer(queue, input, CL_TRUE, 0, NULL, NULL, 0, NULL, NULL);//с хоста

  

    size_t group;
    clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, NULL);
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, NULL, &group, 0, NULL, NULL);
    clFinish(queue);

    //cл57
    clEnqueueReadBuffer(queue, output, CL_TRUE, 0, NULL, NULL, 0, NULL, NULL);//на хост

    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}

