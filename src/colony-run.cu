#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<string.h>

#include<math.h>
#include<limits.h>

// Cuda libraries
#include <cuda.h>
#include <cuda_runtime.h>
// Cuda random number generators
#include <curand.h>
#include <curand_kernel.h>

#include <time.h>
#include <stdlib.h>

double ALPHA = 1;
double BETA = 1;

curandState* DEVSTATES;

double ** EDGE_WEIGHTS;

double ** PHER_TRAILS;

unsigned int ** VISITED;

double * SCORES;

extern int ** SEND_BUF;

extern bool SEND_READY;

extern int ** RECV_BUF;

double BEST_SCORE = -1;

size_t NUM_NODES;
size_t NUM_ANTS;

double INIT_SMALL = 0; //.000001;
double DECAY_RATE = .1;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ double generate(int ind, curandState* dev_states)
{
    // int ind = threadIdx.x;
    curandState localState = dev_states[ind];
    double RANDOM = curand_uniform( &localState );
    dev_states[ind] = localState;
    return RANDOM;
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

// initializes global memory for device 
// inits curand
__global__ void setupKernel(unsigned int seed, curandState* dev_states)
{
    int id = threadIdx.x;
    curand_init ( seed + id, id, 0, &dev_states[id] );
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
    cudaMallocManaged (&DEVSTATES, grid_size * thread_count * sizeof(curandState));
    srand(time(0));
    int seed = rand() + myrank;
    setupKernel<<<1, thread_count>>>(seed, DEVSTATES);
    cudaDeviceSynchronize();

    // create adj matrix for edge weights and phermone trails
    EDGE_WEIGHTS = createEdgeWeightsTSP(nodes, num_coords);
    PHER_TRAILS = createAdjMatrix(num_coords);
    for (int i = 0; i < num_coords; ++i)
    {
        for (int e = 0; e < num_coords; ++e)
        {
            PHER_TRAILS[i][e] = 0;
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

__device__ bool elementOf(unsigned int * visited, size_t size, size_t at)
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
    unsigned int ** visited, 
    double * scores,
    curandState* dev_states,
    size_t num_ants, size_t num_nodes)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    for (; index < num_ants; index += blockDim.x * gridDim.x)
    {
        double total = 0;
        scores[index] = 0;

        // determine what node is next based on probability
        for (size_t step = 0; step < num_nodes; step++)
        {
            if (step == 0)
            {
                // starting city is randomly assigned
                int rand_index = (int) (generate(threadIdx.x, dev_states) * ((num_nodes - 1) + 0.99));
                // print("rand: %f\n",rand_index);
                visited[index][0] = rand_index;
            } else {
                // get the total score of all possible options
                total = 0;
                for (size_t node = 0; node < num_nodes; node++)
                {
                    if (!elementOf(visited[index], step, node))
                    {
                        // printf("step-1: %i node: %i prev: %i index: %i\n", step-1, node, visited[index][step - 1], index);
                        // printf("%f %f\n", pheromone_trails[visited[index][step - 1]][node], edge_weights[visited[index][step - 1]][node]);
                        total += pheromone_trails[visited[index][step - 1]][node] * edge_weights[visited[index][step - 1]][node];
                    }
                }

                double rand = generate(threadIdx.x, dev_states) * total;
                double running_sum = 0;
                for (size_t node = 0; node < num_nodes; node++)
                {
                    if (!elementOf(visited[index], step, node))
                    {
                        running_sum += pheromone_trails[visited[index][step - 1]][node] * edge_weights[visited[index][step - 1]][node];
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


void decayPheromones(double rho, size_t num_nodes)
{
    // decay existing pheromones
    for (int y = 0; y < num_nodes; y++)
    {
        for (int x = 0; x < num_nodes; x++)
        {
            PHER_TRAILS[y][x] *= (1 - rho);
        }
    }
}

// updates the pheromones based on the ant system (Dorigo et al. 1991, Dorigo 1992, Dorigo et al. 1996)
// every ant updates the phermones based on their score
__global__ void updatePheromoneTrailsAS(
    double ** pheromone_trails,
    unsigned int ** visited, 
    double * scores,
    size_t num_ants, size_t num_nodes)
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
    unsigned int ** visited,  double * scores, 
    size_t num_ants, size_t num_nodes
    )
{
    return;
}

// returns the index of the ant with the best score
size_t getBestAnt(size_t num_ants)
{
    double best = SCORES[0];
    size_t idx = 0;
    for (size_t i = 1; i < num_ants; i++)
    {
        if (SCORES[i] < best)
        {
            best = SCORES[i];
            idx = i;
        }
    }
    return idx;
}

void displayAdjMatrix(double ** matrix) {
    for (int y = 0; y < NUM_NODES; ++y)
    {
        for (int x = 0; x < NUM_NODES; x++){
            printf("%lf ", matrix[y][x]);
        }
        printf("\n");
    }
}

void freeCudaAdjMatrix(double ** matrix) {
    for (int y = 0; y < NUM_NODES; ++y)
    {
        cudaFree(matrix[y]);
    }
    cudaFree(matrix);
}

extern "C" void updatePheromones(int num_nodes, int block_count, int thread_count, char * update_rule)
{
    if (strcmp(update_rule, "AS") == 0)
    {
        decayPheromones(.9, num_nodes);
        updatePheromoneTrailsAS<<<block_count, thread_count>>>(PHER_TRAILS, VISITED, SCORES, NUM_ANTS, NUM_NODES);
        cudaDeviceSynchronize();
    }
    else if (strcmp(update_rule, "ACS") == 0)
    {

    }
}

extern "C" void colonyKernelLaunch(size_t num_nodes, size_t num_ants, int block_count, int thread_count)
{
    // Launch the kernel
    colonyKernelTSP<<<block_count,thread_count>>>(PHER_TRAILS, EDGE_WEIGHTS, VISITED, SCORES, DEVSTATES, num_ants, NUM_NODES);
    // cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Calculate the best score and send if better than the best found
    size_t best_idx = getBestAnt(num_ants);
    double best = SCORES[best_idx];
    if (BEST_SCORE == -1 || best < BEST_SCORE)
    {
        printf("Best score: %lf \n", best);
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