#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>

#include<math.h>
#include<float.h>
#include<limits.h>

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

size_t ** VISITED;

double * SCORES;

extern int ** SEND_BUF;

extern bool SEND_READY;

extern int ** RECV_BUF;

double BEST_SCORE = DBL_MAX_EXP;

size_t NUM_NODES;
size_t NUM_ANTS;

double INIT_SMALL = .000001;

__device__ float generate(curandState* globalState, int ind)
{
    // int ind = threadIdx.x;
    curandState localState = globalState[ind %  blockDim.x];
    float RANDOM = curand_uniform( &localState );
    globalState[ind] = localState;
    return RANDOM;
}

__global__ void setupKernel ( curandState * state, unsigned long seed )
{
    int id = threadIdx.x;
    curand_init ( seed + id, id, 0, &state[id] );
}

double ** createAdjMatrix(size_t num_nodes)
{
    double **  m;
    cudaMallocManaged(&m, num_nodes * sizeof(double*));
    for (int i = 0; i < num_nodes; ++i)
    {
        cudaMallocManaged(&m[i], num_nodes * sizeof(double));
    }
    return m;
}

double ** createEdgeWeightsTSP(double ** nodes, int num_nodes)
{   
    double **  weights = createAdjMatrix(num_nodes);

    for (int y = 0; y < num_nodes; ++y)
    {
        for (int x = 0; x < num_nodes; ++x)
        {
            weights[y][x] = 1 / pow(pow(nodes[0][x] - nodes[0][y], 2) + pow(nodes[1][x] - nodes[1][y], 2), .5);
        }
    }
    return weights;
}


// sets up the devices and runs
extern "C" void setupProbelmTSP(int myrank, int grid_size, int thread_count, double ** nodes, size_t num_coords, size_t num_ants)
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
    setupKernel<<<grid_size, thread_count>>>(DEVSTATES,seed);

    EDGE_WEIGHTS = createEdgeWeightsTSP(nodes, num_coords);
    PHER_TRAILS = createAdjMatrix(num_coords);
    for (int i = 0; i < num_coords; ++i)
    {
        for (int e = 0; e < num_coords; ++e)
        {
            PHER_TRAILS[i][e] = INIT_SMALL;
        }
    }

    // create visited array
    // visited[i] is the path of ant i
    cudaMallocManaged(&VISITED, num_ants * sizeof(size_t *));
    for (int i = 0; i < num_ants; i++)
    {
        cudaMallocManaged(&VISITED[i], num_coords * sizeof(size_t *));
    }
    cudaMallocManaged(&SCORES, num_ants * sizeof(double));
    NUM_NODES = num_coords;
    NUM_ANTS = num_ants;
}

// copy memory from device to host
extern "C" void deviceToHost(double * device_pointer, double * host_pointer, unsigned int size)
{
    cudaMemcpy(host_pointer, device_pointer, size, cudaMemcpyDeviceToHost);
}

// copy memory from host to device
extern "C" void hostToDevice(double * host_pointer, double * device_pointer, unsigned int size)
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
__global__ void colonyKernelTSP(
    double ** pheromone_trails, 
    double ** edge_weights,
    size_t ** visited,  double * scores, 
    size_t num_ants, size_t num_nodes, 
    curandState* dev_states)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    for (; index < num_ants; index += blockDim.x)
    {
        double total = 0;
        scores[index] = 0;

        // determine what node is next based on probability
        for (size_t step = 0; step < num_nodes; step++)
        {
            if (step == 0)
            {
                // starting city is randomly assigned
                int rand_index = (int) (generate(dev_states, index) * (num_nodes + 0.999999));
                visited[index][0] = rand_index;
            } else {
                // get the total score of all possible options
                total = 0;
                for (size_t node = 0; node < num_nodes; node++)
                {
                    if (!elementOf(visited[index], step, node))
                    {
                        total += pheromone_trails[visited[index][step]][node] * edge_weights[visited[index][step]][node];
                    }
                }

                double rand = generate(dev_states, index) * total;
                double running_sum = 0;
                for (size_t node = 0; node < num_nodes; node++)
                {
                    if (!elementOf(visited[index], step, node))
                    {
                        running_sum += pheromone_trails[visited[index][step]][node] * edge_weights[visited[index][step]][node];
                        if (running_sum >= rand)
                        {
                            visited[index][step] = node;
                        }
                    }
                }
                // since the edge weights are 1 / distance, score is the sum of distances.
                scores[index] += 1 / edge_weights[visited[index][step-1]][visited[index][step]];
            }
        }
        // add the distance from the last to first node
        scores[index] += 1 / edge_weights[visited[index][num_nodes-1]][visited[index][0]];
    }
}


void decayPheromones(double rho)
{
    // decay existing pheromones
    for (int y = 0; y < NUM_NODES; y++)
    {
        for (int x = 0; x < NUM_NODES; x++)
        {
            PHER_TRAILS[y][x] *= (1 - rho);
        }
    }
}

// updates the pheromones based on the ant system (Dorigo et al. 1991, Dorigo 1992, Dorigo et al. 1996)
// every ant updates the phermones based on their score
__global__ void updatePheromoneTrailsAS(
    double ** pheromone_trails, 
    double ** edge_weights,
    size_t ** visited,  double * scores, 
    size_t num_ants, size_t num_nodes
    )
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    for (; index < num_ants; index += blockDim.x)
    {
        for (size_t step = 0; step < num_nodes; step++)
        {   
            pheromone_trails[visited[index][(num_nodes + step - 1) % num_nodes]][visited[index][step]] += 1 / scores[index];
        }
    }
}

//ant colony system (ACS), introduced by Dorigo and Gambardella (1997)
__global__ void updatePheromoneTrailsACS(
    double ** pheromone_trails, 
    double ** edge_weights,
    size_t ** visited,  double * scores, 
    size_t num_ants, size_t num_nodes
    )
{
    return;
}

void freeCudaAdjMatrix(double ** matrix) {
    for (int y = 0; y < NUM_NODES; ++y)
    {
        cudaFree(matrix[y]);
    }
    cudaFree(matrix);
}

extern "C" void updatePheromones(int block_count, int thread_count, char update_rule)
{
    
}

extern "C" void colonyKernelLaunch(size_t num_nodes, size_t num_ants, int block_count, int thread_count)
{
    // Launch the kernel
    colonyKernelTSP<<<block_count,thread_count>>>(PHER_TRAILS, EDGE_WEIGHTS, VISITED, SCORES, num_ants, NUM_NODES, DEVSTATES);
    cudaDeviceSynchronize();

    // Calculate the best score and send if better than the best found
    double best = DBL_MAX;
    size_t idx;
    for (int i = 0; i < num_ants; i++)
    {
        if (SCORES[i] < best)
        {
            best = SCORES[i];
            idx = i;
        }
    }
    if (best < BEST_SCORE)
    {
        BEST_SCORE = best;
        SEND_READY = true;
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