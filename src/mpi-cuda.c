#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>

unsigned char *g_resultData=NULL;

unsigned char *g_data=NULL;

unsigned char *g_above_row=NULL;

unsigned char *g_below_row=NULL;

extern void HL_initMaster(int pattern, size_t worldWidth, size_t worldHeight, int myrank );

// ----- IMPORT KERNEL FUNCTION ----- //
void HL_kernelLaunch( unsigned char** d_data, unsigned char** d_resultData, 
        //unsigned char* next_above_row, unsigned char* next_below_row, 
        int block_count, int thread_count, 
        unsigned int worldWidth, unsigned int worldHeight, 
        int myrank);

void load_ghost_rows(unsigned char * next_above_row, unsigned char * next_below_row,
    unsigned int worldWidth, unsigned int worldHeight);

void get_send_ghost_rows(unsigned int worldWidth, unsigned int worldHeight);

// ----- IMPORT FREE MEMORY FUNCTION ----- //
extern void freeCudaArrays(int myrank);

// ----- SWAP POINTER FUNCTIONS ----- //
static inline void HL_swapPointers( unsigned char **pA, unsigned char **pB)
{
    // Create temporary holder to hold A's values
    unsigned char *temporary;
    temporary = *pA;

    // Perform the swap
    *pA = *pB;
    *pB = temporary;
}

static inline void HL_printWorld(size_t iteration, size_t g_worldWidth, size_t g_worldHeight)
{
    int i, j;

    printf("Print World - Iteration %lu \n", iteration);
    
    for( i = 1; i < g_worldHeight + 1; i++)
    {
		printf("Row %2d: ", i - 1);
		for( j = 0; j < g_worldWidth; j++)
		{
	    	printf("%u ", (unsigned int)g_data[(i*g_worldWidth) + j]);
		}
		printf("\n");
    }

    printf("\n\n");
}

int main(int argc, char** argv) {
	unsigned int pattern = 0;
	unsigned int worldSize = 0;
	unsigned int iterations = 0;
	unsigned int thread_count = 0;

	pattern = atoi(argv[1]);
	worldSize = atoi(argv[2]);
	iterations = atoi(argv[3]);
	thread_count = atoi(argv[4]);

	//create local storage for top and bottom rows
	g_above_row = calloc( worldSize,sizeof(unsigned char));
	g_below_row = calloc( worldSize,sizeof(unsigned char));
	unsigned char * next_above_row = calloc( worldSize,sizeof(unsigned char));
	unsigned char * next_below_row = calloc( worldSize,sizeof(unsigned char));

	// setput MPI
	MPI_Init(&argc, &argv);

	// Get the number of processes
	int numranks;
	MPI_Comm_size(MPI_COMM_WORLD, &numranks);

	// Get the rank of the process
	int myrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	int block_count = (worldSize * worldSize) / thread_count;

	HL_initMaster(pattern,worldSize, worldSize, myrank);

	double t0, t1;
	if(myrank == 0){
		t0 = MPI_Wtime();
	}

	int aboveRank = ( numranks + myrank - 1 ) % numranks ;
	int belowRank = ( numranks + myrank + 1 ) % numranks ;

	MPI_Status stat;
	MPI_Request send_request_above, recv_request_above, send_request_below, recv_request_below;

	for(int tick = 0; tick < iterations; ++tick) {

		// recieve requests for above and below rows
		MPI_Irecv(next_above_row, worldSize, MPI_CHAR, aboveRank, 'A', MPI_COMM_WORLD, &recv_request_above);
		MPI_Irecv(next_below_row, worldSize, MPI_CHAR, belowRank, 'B', MPI_COMM_WORLD, &recv_request_below);

		// load rows to send from device
		get_send_ghost_rows(worldSize, worldSize);

		// sends for above and below rows
		MPI_Isend(g_below_row, worldSize, MPI_CHAR, belowRank, 'A', MPI_COMM_WORLD, &send_request_above);		
		MPI_Isend(g_above_row, worldSize, MPI_CHAR, aboveRank, 'B', MPI_COMM_WORLD, &send_request_below);

		// wait 
		MPI_Wait(&send_request_below, &stat);
		MPI_Wait(&send_request_above, &stat);
		MPI_Wait(&recv_request_below, &stat);
		MPI_Wait(&recv_request_above, &stat);

		//load recieved rows into device
		load_ghost_rows(next_above_row, next_below_row, worldSize, worldSize);

		// Call the Kernel function
		// HL_kernelLaunch(&g_data, &g_resultData, next_above_row, next_below_row, block_count, thread_count, worldSize, worldSize, myrank);
		HL_kernelLaunch(&g_data, &g_resultData, block_count, thread_count, worldSize, worldSize, myrank);

		// Swap the global data
		HL_swapPointers(&g_data, &g_resultData);
  	}

  	MPI_Barrier( MPI_COMM_WORLD );

	if(myrank == 0){
		t1 = MPI_Wtime();
		printf("\n[WORLD SIZE] %d x %d\n[ITERATIONS] %d\n[EXECUTION TIME] %f\n", worldSize, worldSize, iterations, t1-t0);
		HL_printWorld(iterations, worldSize, worldSize);
	}

	// free cuda memory
	freeCudaArrays(myrank);
	free(g_above_row);
	free(g_below_row);
	free(next_above_row);
	free(next_below_row);

	// Finalize the MPI environment.
	MPI_Finalize();
}
