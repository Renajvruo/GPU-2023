#include <stdio.h>
#include <stdlib.h>
#include <time.h>
void cpu_matrix_mult(int* A, int* B, int* C, int row_A, int col_A, int col_B);
int main()
{
    int row_A, col_A, row_B, col_B;   //矩阵A的行列数 矩阵B的行列数 只有A的列数=B的行数是才能进行矩阵相乘。
    printf("请输入A矩阵行数row_A、列数col_A,B矩阵列数col_B:\n");
    scanf("%d %d %d", &row_A, &col_A, &col_B);
    row_B=col_A;
    // 根据矩阵AB申请内存 一维数组
    int A_Bytes = row_A * col_A;   //矩阵中的数据是int型
    int B_Bytes = row_B * col_B;
    int C_Bytes = row_A * col_B;
    int *A, *B, *C;
    A = (int*)malloc(A_Bytes * sizeof(int));
    B = (int*)malloc(B_Bytes * sizeof(int));
    C = (int*)malloc(C_Bytes * sizeof(int));
    
    // 初始化数据 矩阵AB使用rand函数生成大小10以内的数据 C全部初始化为0(不要了)
    // 为方便CPU 和 GPU 运算结果相互验证，将A B全部初始化为1 C全部初始化为0
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
    //开始矩阵相乘计算
    cpu_matrix_mult(A, B, C, row_A, col_A, col_B);
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
    // 释放内存
    free(A);
    free(B);
    free(C);

    return 0;
}
void cpu_matrix_mult(int* A, int* B, int* C, int row_A, int col_A, int col_B) {
    int i,j,h;
    for (int i = 0; i < row_A; ++i)
    {
        for (int j = 0; j < col_B; ++j)
        {
            int tmp = 0;
            for (int k = 0; k < col_A; ++k)
            {
                tmp += A[i * col_A + k] * B[k * col_B + j];
            }
            C[i * col_B + j] = tmp;
        }
    }
}

