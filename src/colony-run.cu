#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>

// Cuda libraries
#include <cuda.h>
#include <cuda_runtime.h>
// Cuda random number generators
#include <curand.h>
#include <curand_kernel.h>

#include <time.h>
#include <stdlib.h>


__device__ float generate(curandState* globalState, int ind)
{
    //int ind = threadIdx.x;
    curandState localState = globalState[ind];
    float RANDOM = curand_uniform( &localState );
    globalState[ind] = localState;
    return RANDOM;
}

__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
    int id = threadIdx.x;
    curand_init ( seed, id, 0, &state[id] );
}

__global__ void addToCount(int N, int *y, curandState* globalState)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    while (id < blockDim.x * gridDim)
    {
        int number = generate(globalState, id) * 1000000;
        printf("%i\n", number);

        atomicAdd(&(y[0]), number);
        id += blockDim.x * gridDim.x;
    }
}

int main(void)
{
  int thread_count = 5;
  int *y, *d_y;
  y = (int*)malloc(thread_count*sizeof(int));

  cudaMalloc(&d_y, thread_count * sizeof(int));
  cudaMemcpy(d_y, y, thread_count * sizeof(int), cudaMemcpyHostToDevice);

  curandState* devStates;
  cudaMalloc (&devStates, thread_count * sizeof(curandState));
  srand(time(0));
  int seed = rand();

  setup_kernel<<<2, thread_count>>>(devStates,seed);
  addToCount<<<2, thread_count>>>(thread_count, d_y, devStates);

  cudaMemcpy(y, d_y, thread_count * sizeof(int), cudaMemcpyDeviceToHost);
  printf("%i\n", *y);
}


// sets up the devices and runs
extern "C" void setup_and_run(int myrank, grid_size, int thread_count)
{
    // Set device to the rank
    int cudaDeviceCount;
    cudaError_t cE; 
    if( (cE = cudaGetDeviceCount( &cudaDeviceCount)) != cudaSuccess )
    {
        printf(" Unable to determine cuda device count, error is %d, count is %d\n", cE, cudaDeviceCount );
        exit(-1);
    }
    if( (cE = cudaSetDevice( myrank % cudaDeviceCount )) != cudaSuccess )
    {
        printf(" Unable to have myrank %d set to cuda device %d, error is %d \n", myrank, (myrank % cudaDeviceCount), cE);
        exit(-1); 
    }
}

// copy memory from device to host
extern "C" void device_to_host(double * device_pointer, double * host_pointer, unsigned int size)
{
    cudaMemcpy(host_pointer, device_pointer, size, cudaMemcpyDeviceToHost);
}

// copy memory from host to device
extern "C" void host_to_device(double * host_pointer, double * device_pointer, unsigned int size)
{
    cudaMemcpy(device_pointer, host_pointer, size, cudaMemcpyHostToDevice);
}

extern "C" void colony_kernelLaunch( unsigned char** d_data, unsigned char** d_resultData, 
        int block_count, int thread_count, 
        unsigned int worldWidth, unsigned int worldHeight, 
        int myrank){

    // Call the kernel
    // HL_kernel<<<block_count,thread_count>>>(*d_data, *d_resultData, worldWidth, worldHeight);
    cudaDeviceSynchronize();
}


extern "C" void freeCuda(double* ptr){
    cudaFree(ptr);
}