#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<<float.h>

// Cuda libraries
#include <cuda.h>
#include <cuda_runtime.h>
// Cuda random number generators
#include <curand.h>
#include <curand_kernel.h>

#include <time.h>
#include <stdlib.h>

float ALPHA = 1;
float BETA = 1;

curandState* DEVSTATES;

double ** EDGE_WEIGHTS;

double ** PHER_TRAILS;

double ** VISITED;

double * SCORES;

extern int ** SEND_BUF;

extern int ** RECV_BUF;

double BEST_SCORE = DBL_MAX;

size_t NUM_NODES;

__device__ float generate(curandState* globalState, int ind)
{
    // int ind = threadIdx.x;
    curandState localState = globalState[ind %  blockDim.x];
    float RANDOM = curand_uniform( &localState );
    globalState[ind] = localState;
    return RANDOM;
}

__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
    int id = threadIdx.x;
    curand_init ( seed + id, id, 0, &state[id] );
}

double ** create_adj_matrix(size_t num_nodes)
{
    double **  m;
    cudaMallocManaged(&m, num_nodes * sizeof(double*));
    for (int i = 0; i < num_nodes; ++i)
    {
        cudaMallocManaged(&m[i], num_nodes * sizeof(double));
    }
    return m;
}

double ** create_edge_weights_tsp(double ** nodes, int num_nodes)
{   
    double **  weights = create_adj_matrix(num_nodes);

    for (int y = 0; y < num_nodes; ++y)
    {
        for (int x = 0; x < num_nodes; ++x)
        {
            weights[y][x] = 1 / pow(pow(nodes[x][0] - nodes[y][0], 2) + pow(nodes[x][0] - nodes[y][0], 2), .5);
        }
    }
    return weights;
}


// sets up the devices and runs
extern "C" void setup_probelm_tsp(int myrank, int grid_size, int thread_count, double ** nodes, size_t num_coords)
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

    // setup random number generators
    cudaMallocManaged (&DEVSTATES, thread_count * sizeof(curandState));
    srand(time(0));
    int seed = rand() + myrank;
    setup_kernel<<<grid_size, thread_count>>>(DEVSTATES,seed);

    EDGE_WEIGHTS = create_edge_weights_tsp(nodes, num_coords);
    PHER_TRAILS = create_edge_weights_tsp(nodes, num_coords);
    cudaMallocManaged(&VISITED, num_ants * sizeof(size_t *));
    for (int i = 0; i < num_ants; i++)
    {
        cudaMallocManaged(&VISITED[i], num_nodes * sizeof(size_t *));
    }
    cudaMallocManaged(&SCORES, num_ants * sizeof(double));
    NUM_NODES = num_coords;
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

__device__ bool elementOf(size_t * visited, size_t size, size_t at)
{
    for (int i = 0; i < size; ++i)
    {
        if (visited[i] == at)
        {
            return true;
        }
    }
    return false;
}

// each thread will run an ant, who creates one solution to the TSP per ant.
__global__ void colony_kernel(size_t num_ants, size_t num_nodes, size_t ** visited, double * scores)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    for (; index < num_ants; index += blockDim.x)
    {
        double total = 0;
        scores[index] = 0;

        // determine what node is next based on probability
        for (int step = 0; step < num_nodes; step++)
        {
            if (step == 0)
            {
                // starting city is randomly assigned
                int rand_index = (int) (generate(DEVSTATES, index) * (num_nodes + 0.999999));
                visited[index][0] = rand_index;
            } else {
                // get the total score
                total = 0;
                for (int node = 0; node < num_nodes; node++)
                {
                    if (!elementOf(visited[index], step, node))
                    {
                        total += PHER_TRAILS[visited[index][step]][node] * EDGE_WEIGHTS[visited[index][step]][node]
                    }
                }

                double rand = generate(DEVSTATES, index) * total;
                double running_sum = 0;
                for (int node = 0; node < num_nodes; node++)
                {
                    if (!elementOf(visited[index], step, node))
                    {
                        running_sum += PHER_TRAILS[visited[index][step]][node] * EDGE_WEIGHTS[visited[index][step]][node]
                        if (running_sum > rand)
                        {
                            visited[step] = node;
                        }
                    }
                }

                scores[index] += EDGE_WEIGHTS[visited[index][step-1]][visited[index][step]];
            }
        }
    }
}

void freeCudaAdjMatrix(double ** matrix) {
    for (int y = 0; y < NUM_NODES; ++y)
    {
        cudaFree(matrix[y]);
    }
    cudaFree(matrix);
}

extern "C" void colony_kernel_launch(size_t num_nodes, size_t num_ants, int block_count, int thread_count){

    // Launch the kernel
    colony_kernel<<<block_count,thread_count>>>(num_ants, NUM_NODES, VISITED, SCORES);
    cudaDeviceSynchronize();

    // Calculate the best score
    double best = DBL_MAX;
    size_t idx;
    for (int i = 0; i < num_ants; i++)
    {
        if (score[i] < best)
        {
            best = score[i];
            idx = i;
        }
    }
    if (best < BEST_SCORE)
    {
        BEST_SCORE = best;
    }
}

extern "C" void freeCudaGlobal(int num_ants){
    freeCudaAdjMatrix(EDGE_WEIGHTS);
    freeCudaAdjMatrix(PHER_TRAILS);
    for (int i = 0; i < num_ants; i++)
    {
        cudaFree(VISITED[i]);
    }
    cudaFree(VISITED);
    cudaFree(SCORES);
}

extern "C" void freeCuda(double* ptr){
    cudaFree(ptr);
}