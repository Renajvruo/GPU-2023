% % cu
#include <stdio.h>
        __global__ void
        addKernel(int *a, int *b, int *c)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
int main()
{
    int N = 1 << 7;
    int nBytes = N * sizeof(int);
    int *a, *b, *c; // 申请host内存
    a = (int *)malloc(nBytes);
    b = (int *)malloc(nBytes);
    c = (int *)malloc(nBytes);
    for (int i = 0; i < N; ++i)
    { // 初始化数据
        a[i] = i;
        b[i] = 2 * i;
        c[i] = 0;
    }
    int *d_a, *d_b, *d_c; // 申请device内存
    cudaMalloc((void **)&d_a, nBytes);
    cudaMalloc((void **)&d_b, nBytes);
    cudaMalloc((void **)&d_c, nBytes);
    // 将host数据拷贝到device
    cudaMemcpy((void *)d_a, (void *)a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_b, (void *)b, nBytes, cudaMemcpyHostToDevice);
    // 执行kernel
    addKernel<<<1, 128>>>(d_a, d_b, d_c);
    // 将device得到的结果拷贝到host
    cudaMemcpy((void *)c, (void *)d_c, nBytes, cudaMemcpyDeviceToHost);

    // 输出执行结果
    for (int i = 0; i < N; i++)
        printf("a[%d]=%d\tb[%d]=%d\tc[%d]=%d\n", i, a[i], i, b[i], i, c[i]);
    // 释放device内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    // 释放host内存
    free(a);
    free(b);
    free(c);
    return 0;
}