#include <CL/cl.h>
#include <iostream>

using namespace std;

int main() {
    const char * src =
        "__kernel void vec_Change (                 \n"\
        "   __global int *a, __global int *new_a, const unsigned int size) {  \n"\
        "int i = get_global_id (0);                 \n"
        "int N = get_group_id(0);\n"
        "int M = get_local_id(0);\n"
        "printf(\"I am from %d block, %d thread(global index : %d)\", N, M, i); \n"
        "if(i<size)                  \n"\
        "   new_a[i] = a[i] + i;               \n"\
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
    //////////////////////////

    cl_platform_id* platform2 = new cl_platform_id[platformCount];
    clGetPlatformIDs(platformCount, platform2, nullptr);
    for (cl_uint i = 0; i < platformCount; ++i) {
        char platformName[128];
        clGetPlatformInfo(platform2[i], CL_PLATFORM_NAME,
            128, platformName, nullptr);
        std::cout << platformName << std::endl;
    }

    cl_context_properties properties[3] = {
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
        device = devices[0];
    }

    //50
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, NULL);

    size_t srclen[] = { strlen(src) };
    cl_program program = clCreateProgramWithSource(context, 1, &src, srclen, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    //------------------------
    size_t len = 0;
    cout << "succes build " << clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len) << endl;
    //--------------------------------
    cl_kernel kernel = clCreateKernel(program, "vec_Change", NULL);

    //input dATA

    const int mas_size = 400;
    int data[mas_size];
    for (int i = 0; i < mas_size; ++i) {
        data[i] = i;
    }

    int res[mas_size];
    cl_mem input = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * mas_size, NULL, NULL);
    cl_mem output = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * mas_size, NULL, NULL);

    clEnqueueWriteBuffer(queue, input, CL_TRUE, 0, sizeof(int) * mas_size, data, 0, NULL, NULL);

    size_t count = { mas_size };

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);



    size_t group = { 80 };
   // clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, NULL);
    

    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &count, &group, 0, NULL, NULL);
    clFinish(queue);

    //57
    clEnqueueReadBuffer(queue, output, CL_TRUE, 0, sizeof(int) * count, res, 0, NULL, NULL);

    for (int i = 0; i < mas_size; ++i)
        cout << res[i]<<',';

    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}

