#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <cuda.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 32

__global__ void gpu_share_matrix(int* a, int* b, int* c, int m, int n, int k)
{
    __shared__ int sm_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int sm_B[BLOCK_SIZE][BLOCK_SIZE];
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int tmp = 0;    //私有变量临时存储结果
    for (int step = 0; step <= n / BLOCK_SIZE; step++) {//step块的滑动次数
    //从global m 中读数据到 share m，写一次使用多次（share比global快）
        int id_A = step * BLOCK_SIZE + row * n + threadIdx.x;
        if (row < m && BLOCK_SIZE * step + threadIdx.x < n)
            sm_A[threadIdx.y][threadIdx.x] = a[id_A];
        else
            sm_A[threadIdx.y][threadIdx.x] = 0;
        int id_B = step * BLOCK_SIZE * k + col + threadIdx.y * k;
        if (col < k && BLOCK_SIZE * step + threadIdx.y < n)
            sm_B[threadIdx.y][threadIdx.x] = b[id_B];
        else
            sm_B[threadIdx.y][threadIdx.x] = 0;
        __syncthreads();//确保所有线程完成读取
        for (int i = 0; i < BLOCK_SIZE; i++) {
            tmp += sm_A[threadIdx.y][i] * sm_B[i][threadIdx.x];
        }
        __syncthreads();//确保所有线程计算完毕
    }
    if (row < m && col < k)//最后赋给C
        c[row * k + col] = tmp;
}

void cpu_matrix_mult(int* A, int* B, int* C, int M, int N, int K) 
{
    for (int i = 0; i < M; i += BLOCK_SIZE) {
        for (int k = 0; k < N; k += BLOCK_SIZE) { 
            for (int j = 0; j < K; j += BLOCK_SIZE) { 
                    // 分块内的矩阵乘法
                    for (int ii = i; ii < min(i + BLOCK_SIZE, M); ii++) {
                        for (int kk = k; kk < min(k + BLOCK_SIZE, N); kk++) {
                            for (int jj = j; jj < min(j + BLOCK_SIZE, K); jj++) {
                                C[ii * K + jj] += A[ii * N + kk] * B[kk * K + jj];
                            }
                        }
                    }
            }
        }
    }
}

__global__ void gpu_normal_matrix(int* a, int* b, int* c, int m, int n, int k)
{
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    if (row < m && col < k) {//防止溢出的线程工作
        c[row*k+col]=0;
        for (int h = 0; h < n; h++) {
            c[row * k + col] += a[row * n + h] * b[col + k * h];
        }
    }
    return;    
}
int main(int argc, char const* argv[])
{
    int m,n,k;
    m = strtol(argv[1], NULL, 10);
    n = strtol(argv[2], NULL, 10);
    k = strtol(argv[3], NULL, 10);
    int* h_a, * h_b, * h_ccpu, * h_cgs, * h_cgpu;
    cudaMallocHost((void**)&h_a, sizeof(int) * m * n);
    cudaMallocHost((void**)&h_b, sizeof(int) * n * k);
    cudaMallocHost((void**)&h_ccpu, sizeof(int) * m * k);
    cudaMallocHost((void**)&h_cgs, sizeof(int) * m * k);
    cudaMallocHost((void**)&h_cgpu, sizeof(int) * m * k);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = rand() % 5;
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            h_b[i * k + j] = rand() % 5;
        }
    }

    int* d_a, * d_b, * d_cgs, * d_cgpu;
    cudaMalloc((void**)&d_a, sizeof(int) * m * n);
    cudaMalloc((void**)&d_b, sizeof(int) * n * k);
    cudaMalloc((void**)&d_cgs, sizeof(int) * m * k);
    cudaMalloc((void**)&d_cgpu, sizeof(int) * m * k);

    // copy matrix A and B from host to device memory
    cudaMemcpy(d_a, h_a, sizeof(int) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int) * n * k, cudaMemcpyHostToDevice);

    dim3 blockDim,gridDim;
    gridDim.y = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gridDim.x = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blockDim.x =  BLOCK_SIZE;
    blockDim.y =  BLOCK_SIZE;

    cudaEvent_t cudastart;
    cudaEvent_t cudaend;
    cudaEventCreate(&cudastart);
    cudaEventCreate(&cudaend);
    cudaEventRecord(cudastart);
    cudaEventQuery(cudastart);
    gpu_share_matrix << <gridDim, blockDim >> > (d_a, d_b, d_cgs, m, n, k);
    cudaEventRecord(cudaend);
    cudaEventSynchronize(cudaend);
    float ms;
    cudaEventElapsedTime(&ms, cudastart, cudaend);
    printf("GPU share time is %fms\n", ms);
    cudaMemcpy(h_cgs, d_cgs, (sizeof(int) * m * k), cudaMemcpyDeviceToHost);

    cudaEventCreate(&cudastart);
    cudaEventCreate(&cudaend);
    cudaEventRecord(cudastart);
    cudaEventQuery(cudastart);
    gpu_normal_matrix << <gridDim, blockDim >> > (d_a, d_b, d_cgpu, m, n, k);
    cudaEventRecord(cudaend);
    cudaEventSynchronize(cudaend);
    cudaEventElapsedTime(&ms, cudastart, cudaend);
    printf("GPU normal time is %fms\n", ms);
    cudaMemcpy(h_cgpu, d_cgpu, (sizeof(int) * m * k), cudaMemcpyDeviceToHost);

    cpu_matrix_mult(h_a, h_b, h_ccpu, m, n, k);

    int ok_normal = 1;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            if (fabs(h_cgs[i * k + j] - h_ccpu[i * k + j]) > (1.0e-10))
            {

                ok_normal = 0;
                break;
            }
        }
    }
    int ok_share = 1;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            if (fabs(h_cgpu[i * k + j] - h_ccpu[i * k + j]) > (1.0e-10))
            {

                ok_share = 0;
                break;
            }
        }
    }

    if (ok_normal && ok_share)
    {
        printf("Pass!!!\n");
    }
    else
    {
        printf("ok_normal:%d\tok_share:%d\tError!!!\n",ok_normal,ok_share);
    }
    printf("CPU:\n");
    for(int i = 0; i < min(6,m); i++)
    {
        for(int j = 0; j < min(6,k); j++)
        {
            printf("%6d",h_ccpu[i][j]);
        }
        printf("\n");
    }
    printf("GPU global memory:\n");
    for(int i = 0; i < min(6,m); i++)
    {
        for(int j = 0; j < min(6,k); j++)
        {
            printf("%6d",h_cgpu[i][j]);
        }
        printf("\n");
    }
    printf("GPU share memory:\n");
    for(int i = 0; i < min(6,m); i++)
    {
        for(int j = 0; j < min(6,k); j++)
        {
            printf("%6d",h_cgs[i][j]);
        }
        printf("\n");
    }
    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_cgs);
    cudaFree(d_cgpu);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_ccpu);
    cudaFreeHost(h_cgpu);
    cudaFreeHost(h_cgs);
    return 0;
}
