#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
__global__ void gpu_matrix_Kernel(int* d_a, int * d_b, int* d_c, int row_A, int col_A, int col_B)
{
//    printf("%d %d %d\n",row_A, col_A, col_B);
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

// printf("(%d,%d),(%d,%d)\t(%d,%d)\t(%d,%d)\n",gridDim.y,gridDim.x,blockIdx.y,blockIdx.x,threadIdx.y,threadIdx.x, row, col);
    if (row < row_A && col < col_B) {//防止溢出的线程工作
//      printf("(%d,%d),(%d,%d)\t(%d,%d)\t(%d,%d)\n",gridDim.y,gridDim.x,blockIdx.y,blockIdx.x,threadIdx.y,threadIdx.x, row, col);
//      printf("%d %d %d\n",row_A, col_A, col_B);
        d_c[row*col_B+col]=0;
        for (int k = 0; k < col_A; k++) {
            d_c[row * col_B + col] += d_a[row * col_A + k] * d_b[col + col_B * k];
        }
//      printf("%d %d %d\n",row,col,d_c[row*col_B+col]);
    }
    return;
}

int main()
{
    int row_A, col_A, row_B, col_B;   //矩阵A的行列数 矩阵B的行列数 只有A的列数=B的行数是才能进行矩阵相乘。
    printf("请输入A矩阵行数row_A、列数col_A,B矩阵列数col_B:\n");
    scanf("%d %d %d", &row_A, &col_A, &col_B);
    row_B=col_A;
    // 根据矩阵规模申请host内存 一维数组
    int A_Bytes = row_A * col_A;   //矩阵中的数据是int型
    int B_Bytes = row_B * col_B;
    int C_Bytes = row_A * col_B;
    int *A, *B, *C;
    A = (int*)malloc(A_Bytes * sizeof(int));
    B = (int*)malloc(B_Bytes * sizeof(int));
    C = (int*)malloc(C_Bytes * sizeof(int));

    // 初始化数据 矩阵AB使用rand函数生成大小10以内的数据 C全部初始化为0(不要了)
    // 为方便CPU 和 GPU 运算结果相互验证，将A B全部初始化为i++ C全部初始化为0
    int i,j;
    //srand(time(NULL));
    for (i = 0; i < A_Bytes; ++i)
        //A[i] = rand()%10;
        A[i] = i;
    for (i = 0; i < B_Bytes; ++i)
        //B[i] = rand()%10;
        B[i] = i;
    for (i = 0; i < C_Bytes; ++i)
        C[i] = 0;

    //输出A B矩阵
    printf("A矩阵：\n");
    for (i = 0; i < row_A; i++)
    {
        for (j = 0; j < col_A; j++)
        {
            printf("%5d ", A[i*col_A + j]);
        }
        printf("\n");
    }
    printf("B矩阵：\n");
    for (i = 0; i < row_B; i++)
    {
        for (j = 0; j < col_B; j++)
        {
            printf("%5d ", A[i*col_B + j]);
        }
        printf("\n");
    }
    int *d_a, *d_b, *d_c;// 申请device内存
    cudaMalloc((void**)&d_a, A_Bytes * sizeof(int));
    cudaMalloc((void**)&d_b, B_Bytes * sizeof(int));
    cudaMalloc((void**)&d_c, C_Bytes * sizeof(int));
    // 将host数据拷贝到device
    cudaMemcpy((void*)d_a, (void*)A, A_Bytes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_b, (void*)B, B_Bytes * sizeof(int), cudaMemcpyHostToDevice);
    //配置线程
    dim3 blockDim,gridDim;
    printf("请输入blockDim.x和blockDim.y以配置thread\n");
    scanf("%u %u", &blockDim.x, &blockDim.y);
    //blockDim.x = 4;
    //blockDim.y = 4;
    gridDim.x = (col_B + blockDim.x - 1)/blockDim.x;
    gridDim.y = (row_A + blockDim.y - 1)/blockDim.y;
    printf("根据结果矩阵规模和block得出gridDim.x=%u\tgridDim.y=%u\n",gridDim.x,gridDim.y);
    // 执行kernel
    gpu_matrix_Kernel << < gridDim,blockDim >> >(d_a, d_b, d_c, row_A, col_A, col_B);
    // 将device得到的结果拷贝到host
    cudaMemcpy((void*)C, (void*)d_c, C_Bytes * sizeof(int), cudaMemcpyDeviceToHost);
    //输出结果
    printf("C矩阵：\n");
    for (i = 0; i < row_A; i++)
    {
        for (j = 0; j < col_B; j++)
        {
            printf("%5d ", C[i*col_B + j]);
        }
        printf("\n");
    }
    // 释放device内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    // 释放host内存
    free(A);
    free(B);
    free(C);
    return 0;
}