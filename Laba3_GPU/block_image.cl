#define SCALE_BLOCKS 16


__kernel void multi_block(__global float *a, __global float *b, __global float *c, int n) {
    __local float A[SCALE_BLOCKS][SCALE_BLOCKS];
    __local float B[SCALE_BLOCKS][SCALE_BLOCKS];
    int loc_row = get_local_id(0);
    int loc_col = get_local_id(1);
    int glob_row = get_global_id(0);
    int glob_col = get_global_id(1);
    int blocks = n / SCALE_BLOCKS;
    float summa = 0;
    for (int i = 0; i < blocks; i++) {
        A[loc_col][loc_row] = a[glob_col * n + SCALE_BLOCKS * i + loc_row];
        B[loc_col][loc_row] = b[(SCALE_BLOCKS * i + loc_col) * n + glob_row];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j = 0; j < SCALE_BLOCKS; j++)
            summa += A[loc_col][j] * B[j][loc_row];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[glob_col * n + glob_row] = summa;
}

__kernel void multi_image(__read_only image2d_t a, __read_only image2d_t b, __write_only image2d_t c, int n) {
    __local float A[SCALE_BLOCKS][SCALE_BLOCKS];
    __local float B[SCALE_BLOCKS][SCALE_BLOCKS];
    int loc_row = get_local_id(0);
    int loc_col = get_local_id(1);
    int glob_row = get_global_id(0);
    int glob_col = get_global_id(1);
    int blocks = n / SCALE_BLOCKS;
    float summa = 0.0;
    for (int i = 0; i < blocks; i++) {
        float x = read_imagef(a, (int2)(SCALE_BLOCKS * i + loc_row, glob_col)).x;
        float y = read_imagef(b, (int2)(glob_row, SCALE_BLOCKS * i + loc_col)).x;
        A[loc_col][loc_row] = x;
        B[loc_col][loc_row] = y;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j = 0; j < SCALE_BLOCKS; j++)
            summa += A[loc_col][j] * B[j][loc_row];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    write_imagef(c, (int2)(glob_row, glob_col), summa);
}

