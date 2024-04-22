#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>

size_t * SEND_BUF = NULL;
size_t * RECV_BUF = NULL;
bool SEND_READY = false;

double RHO = .1;

// external cuda functions
void freeCudaGlobal(int num_ants);
void setupProbelmTSP(int myrank, int grid_size, int thread_count, double ** nodes, size_t num_coords, size_t num_ants);
void colonyKernelLaunch(size_t num_nodes, size_t num_ants, int block_count, int thread_count);
void updatePheromones(int num_nodes, int block_count, int thread_count, char * update_rule, bool decay, double rho);

// Creates array of coordinates from file. 
// File should consist of comma separated values x,y per line per coordinates. No more than 128 characters per line
double ** interpret_file(char * filename)
{
	FILE* ptr = fopen(filename, "r");
	if (NULL == ptr) {
        printf("file can't be opened \n");
        return NULL;
    }

    char * ch, * p;
    char line[128];

    double **  data = (double **) malloc(2 * sizeof(double*));
    data[0] = calloc(0, sizeof(double *));
    data[1] = calloc(0, sizeof(double *));
    size_t size = 0;

    while (fgets(line, 128, ptr))
    {
    	p = strtok(line, ",");
    	data[0] = (double *) realloc(data[0], sizeof(double) * (size + 1));
    	data[1] = (double *) realloc(data[1], sizeof(double) * (size + 1));
    	int i = 0;

	    while(p!=NULL)
	    {
	        data[i++][size] = atof(p);
	        p = strtok(NULL, ",");
	    }
	    size++;
    }

    return data;
}

void outputResults(char * filename, double ** coords, size_t num_coords, size_t * order)
{

	FILE* fptr = fopen(filename, "w");
	if (NULL == fptr) {
        printf("file can't be opened \n");
        return;
    }

    for (int i = 0; i < num_coords; i++)
	{
    	fprintf(fptr, "%lf,%lf\n", coords[0][order[i]], coords[1][order[i]]);
    }
}

// gets the number of points in the file
size_t get_num_points(char * filename)
{
	FILE* ptr = fopen(filename, "r");
	if (NULL == ptr) {
        printf("file can't be opened \n");
        return 0;
    }

    size_t size = 0;
    char line[128];
    while (fgets(line, 128, ptr))
    {
	    size++;
    }
    return size;
}

int main(int argc, char** argv) {
	// MPI STUFF 
	// setput MPI
	MPI_Init(&argc, &argv);

	// Get the number of processes
	int numranks;
	MPI_Comm_size(MPI_COMM_WORLD, &numranks);

	// Get the rank of the process
	int myrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	double t0, t1;
	if(myrank == 0){
		t0 = MPI_Wtime();
	}

	// extract parameters
	if (argc != 5)
	{
		printf("Incorrect number of provided parameters: %i, expected 5.\n", argc);
		printf("Use -$ <total ants> <iterations> <thread_count> <input file name> \n");
		return 0;
	}
	unsigned int colonies = 0;
	unsigned int total_ants = 0;
	unsigned int iterations = 0;
	unsigned int thread_count = 0;
	char * filename;

	colonies = numranks; // The number of colonies is the number of MPI ranks
	total_ants = atoi(argv[1]);
	iterations = atoi(argv[2]);
	thread_count = atoi(argv[3]);
	filename = argv[4];

	// extract points from file and distribute to all ranks
	size_t num_coords = 0;
	double ** coords;
	if (myrank == 0) {
		coords = interpret_file(filename);
		num_coords = get_num_points(filename);
	} 

	MPI_Bcast(&num_coords, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	if (myrank != 0) {
		coords = (double **) malloc(2 * sizeof(double*));
		coords[0] = (double *) malloc(num_coords * sizeof(double));
		coords[1] = (double *) malloc(num_coords * sizeof(double));
	}

	// send the file data to all ranks
	MPI_Bcast(coords[0], num_coords, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(coords[1], num_coords, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	// set up colony for this process (create ants, init pheromone trails)
	int ants_per_colony = (total_ants + colonies - 1) / colonies;
	int blocks_per_grid = (ants_per_colony + thread_count - 1) / thread_count;
	setupProbelmTSP(myrank, blocks_per_grid, thread_count, coords,  num_coords, ants_per_colony);

	// set up MPI reception buffer for processing asynchronous communication
	RECV_BUF = malloc(num_coords * sizeof(size_t));
	SEND_BUF = malloc(num_coords * sizeof(size_t));
	MPI_Request recv_request, send_request;
	MPI_Irecv(RECV_BUF, num_coords, MPI_UNSIGNED_LONG, MPI_ANY_SOURCE,  MPI_ANY_TAG, MPI_COMM_WORLD, &recv_request);

	MPI_Status stat;
	// execution loop (cuda for processing), MPI for communicating best solutions
	for (int i = 0; i < iterations; ++i)
	{
		// launch kernel for one iteration
		colonyKernelLaunch(num_coords, ants_per_colony, blocks_per_grid, thread_count);

		// check if new best solution is found
		if (SEND_READY)
		{
			printf("Best best solution found at %i \n", myrank);
			SEND_READY = false;

			// distribute solution to all other colonies
			// potentially other methods to distribute only certain colonies
			for (int rank = 0; rank < numranks; ++rank){
				if (rank != myrank)
				{
					MPI_Isend(SEND_BUF, num_coords, MPI_UNSIGNED_LONG, rank, 'G', MPI_COMM_WORLD, &send_request);
				}
			}
		}

		//check if there are any messages waiting
		int flag = 0;
		MPI_Test(&recv_request, &flag, &stat);
		if (flag)
		{
			// update pheromones based on recieved message
			updatePheromones(num_coords, blocks_per_grid, thread_count, "MESSAGE", false, RHO);

			// post another recieve request
			MPI_Irecv(RECV_BUF, num_coords, MPI_UNSIGNED_LONG, MPI_ANY_SOURCE,  MPI_ANY_TAG, MPI_COMM_WORLD, &recv_request);
		}

		// update pheromones
		updatePheromones(num_coords, blocks_per_grid, thread_count, "AS", true, RHO);
	}

	// synchronize ranks
	MPI_Barrier( MPI_COMM_WORLD );

	//output results
	if (myrank == 0)
	{
		// for (int i = 0; i < num_coords; i++)
		// {
		// 	printf("%lf,%lf\n", coords[0][SEND_BUF[i]], coords[1][SEND_BUF[i]]);
		// }

		outputResults("output.txt", coords, num_coords, SEND_BUF);
	}

	// free memory
	free(coords[0]);
	free(coords[1]);
	free(coords);
	free(RECV_BUF);
	free(SEND_BUF);

	// free cuda memory
	freeCudaGlobal(ants_per_colony);

	// Finalize the MPI environment.
	MPI_Finalize();
}
