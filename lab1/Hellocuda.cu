% % cu
#include <stdio.h>
    int
    main()
{
    int number;
    cudaGetDeviceCount(&number);
    printf("共有%d台设备:\n", number);
    for (int i = 0; i < number; ++i)
    {
        cudaDeviceProp Dev;
        cudaGetDeviceProperties(&Dev, i);
        printf("第%d台设备\n", i);
        printf("设备名称(name): %s\n", Dev.name);
        printf("计算能力(compute capability): %d.%d\n", Dev.major, Dev.minor);
        printf("设备可用全局内存总量(total global mem) : %fMB, %llubytes\n", (float)Dev.totalGlobalMem / 1024 / 1024, (unsigned long long)Dev.totalGlobalMem);
        printf("每线程块最大线程数(max threads per block): %d\n", Dev.maxThreadsPerBlock);
        printf("每线程块可用共享内存容量(shared mem per block): %f KB, %lu bytes \n", (float)Dev.sharedMemPerBlock / 1024, Dev.sharedMemPerBlock);
        printf("每线程块可用的32位寄存器数量(regs per block): %d \n", Dev.regsPerBlock);
        printf("每个处理器簇最大驻留线程数(max Threads Per MultiProcessor): %d\n", Dev.maxThreadsPerMultiProcessor);
        printf("设备中的处理器簇数量(multiprocessor count): %d\n", Dev.multiProcessorCount);
    }
}