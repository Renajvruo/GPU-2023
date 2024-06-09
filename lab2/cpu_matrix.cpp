#include <stdio.h>
#include <stdlib.h>
#include <time.h>
void cpu_matrix_mult(int* A, int* B, int* C, int row_A, int col_A, int col_B);
int main()
{
    int row_A, col_A, row_B, col_B;   //����A�������� ����B�������� ֻ��A������=B�������ǲ��ܽ��о�����ˡ�
    printf("������A��������row_A������col_A,B��������col_B:\n");
    scanf("%d %d %d", &row_A, &col_A, &col_B);
    row_B=col_A;
    // ���ݾ���AB�����ڴ� һά����
    int A_Bytes = row_A * col_A;   //�����е�������int��
    int B_Bytes = row_B * col_B;
    int C_Bytes = row_A * col_B;
    int *A, *B, *C;
    A = (int*)malloc(A_Bytes * sizeof(int));
    B = (int*)malloc(B_Bytes * sizeof(int));
    C = (int*)malloc(C_Bytes * sizeof(int));
    
    // ��ʼ������ ����ABʹ��rand�������ɴ�С10���ڵ����� Cȫ����ʼ��Ϊ0(��Ҫ��)
    // Ϊ����CPU �� GPU �������໥��֤����A Bȫ����ʼ��Ϊ1 Cȫ����ʼ��Ϊ0
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
        
    //���A B���� 
    printf("A����\n"); 
    for (i = 0; i < row_A; i++)
    {
        for (j = 0; j < col_A; j++)
        {
            printf("%5d ", A[i*col_A + j]);
        }
        printf("\n");
    }
    printf("B����\n"); 
    for (i = 0; i < row_B; i++)
    {
        for (j = 0; j < col_B; j++)
        {
            printf("%5d ", A[i*col_B + j]);
        }
        printf("\n");
    }
    //��ʼ������˼���
    cpu_matrix_mult(A, B, C, row_A, col_A, col_B);
    //������
    printf("C����\n");
    for (i = 0; i < row_A; i++)
    {
        for (j = 0; j < col_B; j++)
        {
            printf("%5d ", C[i*col_B + j]);
        }
        printf("\n");
    }
    // �ͷ��ڴ�
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

