#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include <cuda.h>
#include <cuda_runtime.h>



// copy memory from device to host
extern "C" void host_to_device(float * device_pointer, float * host_pointer unsigned int size)
{
    cudaMemcpy(host_pointer, device_pointer, size, cudaMemcpyDeviceToHost);
}

// copy memory from host to device
extern "C" void host_to_device(float * host_pointer, float * device_pointer unsigned int size)
{
    cudaMemcpy(device_pointer, host_pointer, size, cudaMemcpyHostToDevice);
}

extern "C" void HL_kernelLaunch( unsigned char** d_data, unsigned char** d_resultData, 
        int block_count, int thread_count, 
        unsigned int worldWidth, unsigned int worldHeight, 
        int myrank){

    // Call the kernel
    HL_kernel<<<block_count,thread_count>>>(*d_data, *d_resultData, worldWidth, worldHeight);
    cudaDeviceSynchronize();
}


extern "C" void freeCuda(float* ptr){
    cudaFree(ptr);
}