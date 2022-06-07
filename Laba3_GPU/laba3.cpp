#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <CL/cl.h>
#include <omp.h>
#include <limits>
#include <fstream>

using namespace std;

string readFile(const std::string &path_to_file) {
    std::ifstream file(path_to_file);
    std::string text, str;
    while (std::getline(file, str)) {
        text += str;
        text.push_back('\n');
    }
    return text;
}

template <typename T>
void write_value_in_Matrix(std::vector<T> &mas) {
    std::random_device some_device;
    std::mt19937 mrs(some_device());
    std::uniform_real_distribution<> urd(-100.0, 100.0);
    size_t size = mas.size();
    for (T &element : mas)
        element = urd(mrs);
}

bool comparison(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); i++)
        if (std::abs(a[i] - b[i]) >= 1e-4f)
            return false;
    return true;
}

void multiply_cpu(float *a, float *b, float *c, int n) {
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            for (int i = 0; i < n; i++) {
                c[row * n + i] += a[row * n + col] * b[i + col * n];

            }
        }
    }
}

void multiply_omp1(float *a, float *b, float *c, int n) {
#pragma omp parallel for
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            for (int i = 0; i < n; i++) {
                c[row * n + i] += a[row * n + col] * b[i + col * n];

            }
        }
    }
}


void standart_multiply(float *a, float *b, float *c, int n,  cl_device_id device) {
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    const char * standart_multiply =
        "__kernel void multiply(__global float *a, __global float *b, __global float *c, int n) {\n"\
        "int row = get_global_id(1);\n"\
        "int col = get_global_id(0);\n"\
        "float summa = 0.0;\n"\
        "int i;\n"\
        "for (i = 0; i < n; i++)\n"\
        "   summa += a[row * n + i] * b[col + n * i];\n"\
        "c[n * row + col] = summa;\n"\
        "}\n";

    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, NULL);

    size_t srclen[] = { strlen(standart_multiply) };
    cl_program program = clCreateProgramWithSource(context, 1, &standart_multiply, srclen, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    //------------------------
    size_t len = 0;
    std::cout << "\nsucces build " << clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len) << std::endl;

    cl_kernel kernel = clCreateKernel(program, "multiply", NULL);

    //input dATA

    cl_mem aBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n * n, NULL, NULL);
    cl_mem bBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n * n, NULL, NULL);
    cl_mem cBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n * n, NULL, NULL);



    clEnqueueWriteBuffer(queue, aBuf, CL_TRUE, 0, sizeof(float) * n * n, a, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bBuf, CL_TRUE, 0, sizeof(float) * n * n, b, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, cBuf, CL_TRUE, 0, sizeof(float) * n * n, c, 0, NULL, NULL);

    //size_t count = { n };
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &aBuf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bBuf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &cBuf);
    clSetKernelArg(kernel, 3, sizeof(int), &n);

    size_t glob_work_size[] = { static_cast<size_t>(n), static_cast<size_t>(n) };
    size_t loc_work_size[] = { 16, 16 };

    double begin = omp_get_wtime();
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, glob_work_size, loc_work_size, 0, NULL, NULL);
    clFinish(queue);
    double end = omp_get_wtime();
    std::cout << "GPU VERS standart" << "\n" << (end - begin) << std::endl;



    clEnqueueReadBuffer(queue, cBuf, CL_TRUE, 0, sizeof(float) * n * n, c, 0, NULL, NULL);

    clReleaseMemObject(aBuf);
    clReleaseMemObject(bBuf);
    clReleaseMemObject(cBuf);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

void block_multiply(float *a, float *b, float *c, int n, cl_device_id device) {
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    /*  const char * block_multiply =
          "int BLOCK_SIZE = 16;\n"\
          "__kernel void block_multi(__global float *a, __global float *b, __global float *c, int m, int n, int k) {\n"\
          "__local float A[BLOCK_SIZE][BLOCK_SIZE];\n"\
          "__local float B[BLOCK_SIZE][BLOCK_SIZE];\n"\
          "int local_row = get_local_id(0);\n"\
          "int local_col = get_local_id(1);\n"\
          "int row = get_global_id(0);\n"\
          "int col = get_global_id(1);\n"\
          "int blocks = n / BLOCK_SIZE;\n"\
          "float s = 0.0;\n"\
          "for (int i = 0; i < blocks; i++) {\n"\
          "    A[local_col][local_row] = a[col * n + BLOCK_SIZE * i + local_row];\n"\
          "    B[local_col][local_row] = b[(BLOCK_SIZE * i + local_col) * n + row];\n"\
          "    barrier(CLK_LOCAL_MEM_FENCE);\n"\
          "    for (int j = 0; j < BLOCK_SIZE; j++)\n"\
          "        s += A[local_col][j] * B[j][local_row];\n"\
          "    barrier(CLK_LOCAL_MEM_FENCE);\n"\
          "}\n"\
          "c[col * k + row] = s;\n"\
          "}\n";*/

          /*const char * block_multiply =
                  "__kernel void block_multi(__global float *a, __global float *b, __global float *c, int m, int n, int k) {\n"
                  "int BLOCK_SIZE = 16;\n"\
                  "__local float A[BLOCK_SIZE][BLOCK_SIZE];\n"
                  "__local float B[BLOCK_SIZE][BLOCK_SIZE];\n"
                  "int local_row = get_local_id(1);\n"
                  "int local_col = get_local_id(0);\n"
                  "int row = get_global_id(1);\n"
                  "int col = get_global_id(0);\n"
                  "int blocks = n / BLOCK_SIZE;\n"
                  "float s = 0;\n"
                  "for (int i = 0; i < blocks; i++) {\n"
                  "   A[local_row][local_col] = a[row * n + BLOCK_SIZE * i + local_col];\n"
                  "   B[local_row][local_col] = b[(BLOCK_SIZE * i + local_row) * n + col];\n"
                  "   barrier(CLK_GLOBAL_MEM_FENCE);\n"
                  "   for (int j = 0; j < BLOCK_SIZE; j++)\n"
                  "       s += A[local_row][j] * B[j][local_col];\n"
                  "   barrier(CLK_GLOBAL_MEM_FENCE);\n"
                  "}\n"
                  "c[row * n + col] = s;\n"
                  "}\n";*/

             

    //---------------------------
    std::string src = readFile("C:\\Users\\HP-PC\\Desktop\\4 курс\\программирование на новых арх\\laba1\\GPU_laba1\\Laba3_GPU\\block_image.cl");
    const char *str[] = { src.c_str() };
    /////////////////////////

    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, NULL);

    cl_program program = clCreateProgramWithSource(context, 1, str, nullptr, nullptr);


    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    //------------------------
    size_t len = 0;
    std::cout << "\nsucces build " << clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len) << std::endl;

    
    cl_kernel kernel = clCreateKernel(program, "multi_block", NULL);

    //input dATA

    cl_mem aBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n * n, NULL, NULL);
    cl_mem bBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n * n, NULL, NULL);
    cl_mem cBuf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n * n, NULL, NULL);



    clEnqueueWriteBuffer(queue, aBuf, CL_TRUE, 0, sizeof(float) * n * n, a, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bBuf, CL_TRUE, 0, sizeof(float) * n * n, b, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, cBuf, CL_TRUE, 0, sizeof(float) * n * n, c, 0, NULL, NULL);

    //size_t count = { n };
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &aBuf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bBuf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &cBuf);
    clSetKernelArg(kernel, 3, sizeof(int), &n);

    
    size_t glob_work_size[] = { static_cast<size_t>(n), static_cast<size_t>(n) };
    size_t loc_work_size[] = { 16, 16 };

    double begin = omp_get_wtime();
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, glob_work_size, loc_work_size, 0, NULL, NULL);
    clFinish(queue);
    double end = omp_get_wtime();
    std::cout << "GPU VERS block" << "\n" << (end - begin) << std::endl;



    clEnqueueReadBuffer(queue, cBuf, CL_TRUE, 0, sizeof(float) * n * n, c, 0, NULL, NULL);

    clReleaseMemObject(aBuf);
    clReleaseMemObject(bBuf);
    clReleaseMemObject(cBuf);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}


void image_multiply(float *a, float *b, float *c, int n, cl_device_id device) {
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    /*const char * image_multiply =
        "__kernel void multiplyImage(__read_only image2d_t a, __read_only image2d_t b, __write_only image2d_t c, int m, int n, int k) { \n"\
        "int BLOCK_SIZE = 16; \n"\
        "__local float A[BLOCK_SIZE][BLOCK_SIZE]; \n"\
        "__local float B[BLOCK_SIZE][BLOCK_SIZE]; \n"\
        "int local_row = get_local_id(0);\n"\
        "int local_col = get_local_id(1);\n"\
        "int row = get_global_id(0);\n"\
        "int col = get_global_id(1);\n"\
        "int blocks = n / BLOCK_SIZE;\n"\
        "float s = 0;\n"\
        "for (int i = 0; i < blocks; i++) { \n"\
        "    float x = read_imagef(a, (int2)(BLOCK_SIZE * i + local_row, col)).x; \n"\
        "    float y = read_imagef(b, (int2)(row, BLOCK_SIZE * i + local_col)).x; \n"\
        "    A[local_col][local_row] = x;\n"\
        "    B[local_col][local_row] = y;\n"\
        "    barrier(CLK_LOCAL_MEM_FENCE); \n"\
        "    for (int j = 0; j < BLOCK_SIZE; j++) {\n"\
        "        s += A[local_col][j] * B[j][local_row];\n"\
        "    }\n"\
        "    barrier(CLK_LOCAL_MEM_FENCE);\n"\
        "}\n"\
        "write_imagef(c, (int2)(row, col), s);\n"\
        "}\n";*/

        //---------------------------
    std::string src = readFile("C:\\Users\\HP-PC\\Desktop\\4 курс\\программирование на новых арх\\laba1\\GPU_laba1\\Laba3_GPU\\block_image.cl");
    const char *str[] = { src.c_str() };
    /////////////////////////


    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, NULL);

    cl_program program = clCreateProgramWithSource(context, 1, str, nullptr, nullptr);


    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    //------------------------
    size_t len = 0;
    std::cout << "\nsucces build " << clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len) << std::endl;

    cl_kernel kernel = clCreateKernel(program, "multi_image", NULL);

    //input dATA
    cl_image_format format;
    format.image_channel_order = CL_R;
    format.image_channel_data_type = CL_FLOAT;
    size_t beginning[] = { 0, 0, 0 };


    cl_mem aBuf = clCreateImage2D(context, CL_MEM_READ_WRITE, &format, static_cast<size_t>(n), static_cast<size_t>(n), 0, nullptr, nullptr);
    cl_mem bBuf = clCreateImage2D(context, CL_MEM_READ_WRITE, &format, static_cast<size_t>(n), static_cast<size_t>(n), 0, nullptr, nullptr);
    cl_mem cBuf = clCreateImage2D(context, CL_MEM_READ_WRITE, &format, static_cast<size_t>(n), static_cast<size_t>(n), 0, nullptr, nullptr);

    {
        size_t square[] = { static_cast<size_t>(n), static_cast<size_t>(n), 1 };
        clEnqueueWriteImage(queue, aBuf, CL_TRUE, beginning, square, 0, 0, a, 0, nullptr, nullptr);
    }
    {
        size_t square[] = { static_cast<size_t>(n), static_cast<size_t>(n), 1 };
        clEnqueueWriteImage(queue, bBuf, CL_TRUE, beginning, square, 0, 0, b, 0, nullptr, nullptr);
    }

    //size_t count = { n };
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &aBuf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bBuf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &cBuf);
    clSetKernelArg(kernel, 3, sizeof(int), &n);

    size_t glob_work_size[] = { static_cast<size_t>(n), static_cast<size_t>(n) };
    size_t loc_work_size[] = { 16, 16 };

    double begin = omp_get_wtime();
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, glob_work_size, loc_work_size, 0, NULL, NULL);
    clFinish(queue);
    double end = omp_get_wtime();
    std::cout << "GPU VERS image" << "\n" << (end - begin) << std::endl;


    {
        size_t square[] = { static_cast<size_t>(n), static_cast<size_t>(n), 1 };
        clEnqueueReadImage(queue, cBuf, CL_TRUE, beginning, square, 0, 0, c, 0, nullptr, nullptr);
    }

    clReleaseMemObject(aBuf);
    clReleaseMemObject(bBuf);
    clReleaseMemObject(cBuf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}





int main() {


//////    ---CL
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

    constexpr int n = 16*100;


    std::vector<float> a(n * n);
    std::vector<float> b(n * n);
    std::vector<float> cTotal(n * n);

    write_value_in_Matrix(a);
    write_value_in_Matrix(b);


    {
        double begin = omp_get_wtime();
        multiply_cpu(a.data(), b.data(), cTotal.data(), n);
        double end = omp_get_wtime();
        std::cout << "Sequential: " << (end - begin) << std::endl;
    }

    {
        std::vector<float> c(n * n);
        double begin = omp_get_wtime();
        multiply_omp1(a.data(), b.data(), c.data(), n);
        double end = omp_get_wtime();
        std::cout << "OpenMP: " << (end - begin) << ' ';
        if (comparison(c, cTotal) == true)
            cout << "correct result" << endl;
    }

    {
        std::vector<float> c(n * n, 0);
        standart_multiply(a.data(), b.data(), c.data(), n, device);
        if (comparison(c, cTotal) == true)
            cout << "correct result" << endl;
    }

    {
        std::vector<float> c(n * n, 0);
        block_multiply(a.data(), b.data(), c.data(), n, device);
        if (comparison(c, cTotal) == true)
            cout << "correct result" << endl;
    }

    {
        std::vector<float> c(n * n, 0);
        image_multiply(a.data(), b.data(), c.data(), n, device);
        if (comparison(c, cTotal) == true)
            cout << "correct result" << endl;
    }



}