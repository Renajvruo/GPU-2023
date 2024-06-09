#include <cuda.h>
#include <stdio.h>
#include <string.h>

__global__  void what_is_my_id(int N, unsigned int * const block,
  unsigned int * const thread,
  unsigned int * const warp,
  unsigned int * const calc_thread,
  long int * const completionTime) 
{
    //计算线程(全局)索引
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    block[thread_idx] = blockIdx.x;   //保存线程在块内索引
    thread[thread_idx] = threadIdx.x; //保存该块在grid中的索引
    warp[thread_idx] = threadIdx.x >> 5;  //保存线程所在的线程束
    calc_thread[thread_idx] = thread_idx; //保存线程索引
    completionTime[thread_idx] = clock(); //记录该线程结束的时间，用于后续确认哪些线程属于一个warp
}

void show_result(int N, unsigned int * const block,
    unsigned int * const thread,
    unsigned int * const warp,
    unsigned int * const calc_thread, 
    long int * const completionTime); 

void get_time(long int * completionTime, int N);

int main()
{

  //配置线程
  unsigned int grid_size, block_size;
  printf("请输入grid和block的大小(默认一维):\n");
  scanf("%u %u", &grid_size, &block_size);
  const int N = grid_size * block_size;
  size_t size = N * sizeof(unsigned int);
  size_t size_time = N * sizeof(long int);

  //申请host内存 以存储每个线程的信息
  unsigned int *h_block;
  unsigned int *h_thread;
  unsigned int *h_warp;
  unsigned int *h_calc_thread;
  long int *h_completionTime;

  h_block = (unsigned int*)malloc(size);
  h_thread = (unsigned int*)malloc(size);
  h_warp = (unsigned int*)malloc(size);
  h_calc_thread = (unsigned int*)malloc(size);
  h_completionTime = (long int*)malloc(size_time);

  //申请device内存
  unsigned int *d_block;
  unsigned int *d_thread;
  unsigned int *d_warp;
  unsigned int *d_calc_thread;
  long int *d_completionTime;

  cudaMallocManaged((void**)&d_block, size);
  cudaMallocManaged((void**)&d_thread, size);
  cudaMallocManaged((void**)&d_warp, size);
  cudaMallocManaged((void**)&d_calc_thread, size);
  cudaMallocManaged((void**)&d_completionTime, size_time);

  //调用kernel函数,配置是之前输入的
  what_is_my_id<<<grid_size, block_size>>>(N, d_block, d_thread, d_warp, d_calc_thread, d_completionTime);

  // 将device得到的结果拷贝到host
  cudaMemcpy((void*)h_block, (void*)d_block, size, cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)h_thread, (void*)d_thread, size, cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)h_warp, (void*)d_warp, size, cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)h_calc_thread, (void*)d_calc_thread, size, cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)h_completionTime, (void*)d_completionTime, size_time, cudaMemcpyDeviceToHost);

  //处理clock返回的时间
  get_time(h_completionTime, N);
  //输出结果
  show_result(N, h_block, h_thread, h_warp, h_calc_thread, h_completionTime);
  //释放内存
  free(h_block);
  free(h_thread);
  free(h_warp);
  free(h_calc_thread);
  free(h_completionTime);
  cudaFree(d_block);
  cudaFree(d_thread);
  cudaFree(d_warp);
  cudaFree(d_calc_thread);
  cudaFree(d_completionTime);
}

void get_time(long int * completionTime, int N)
{
    int i;
    long min_time = 1 << 31 - 1 ;
    for(i = 0; i < N; i++)
        min_time = min_time < completionTime[i] ? min_time : completionTime[i];
    for(i = 0; i < N; i++)
        completionTime[i] -= min_time;
    return;
}
void show_result(int N, unsigned int * const block,
    unsigned int * const thread,
    unsigned int * const warp,
    unsigned int * const calc_thread, 
    long int * const completionTime) 
{
    printf("calc_thread\t\tblock\t\tthread\t\twarp\t\tcompletionTime\n");
    int i;
    for(i = 0; i < N; i++)
    {
        printf("\t%u\t\t%u\t\t%u\t\t%u\t\t%ld\n",calc_thread[i], block[i], thread[i], warp[i], completionTime[i]);
    }
}