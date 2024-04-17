#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>

int * SEND_BUF = NULL;
int * RECV_BUF = NULL;
bool SEND_READY = false;

// external cuda functions
void freeCudaGlobal(int num_ants);
void setupProbelmTSP(int myrank, int grid_size, int thread_count, double ** nodes, size_t num_coords, size_t num_ants);
void colonyKernelLaunch(size_t num_nodes, size_t num_ants, int block_count, int thread_count);

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

    printf("%i | %lu | (%f, %f)\n", myrank, num_coords, coords[0][0], coords[1][0]);
    printf("%i | %lu | (%f, %f)\n", myrank, num_coords, coords[0][1], coords[1][1]);

	// set up colony for this process (create ants, init pheromone trails)
	int ants_per_colony = (total_ants + colonies - 1) / colonies;
	setupProbelmTSP(myrank, (ants_per_colony + thread_count - 1) / thread_count, thread_count, coords,  num_coords, ants_per_colony);

	// set up MPI reception buffer for processing asynchronous communication
	RECV_BUF = malloc(num_coords * sizeof(size_t));
	SEND_BUF = malloc(num_coords * sizeof(size_t));
	MPI_Request recv_request;
	MPI_Irecv(RECV_BUF, num_coords, MPI_UNSIGNED_LONG, MPI_ANY_SOURCE,  MPI_ANY_TAG, MPI_COMM_WORLD, &recv_request);

	// execution loop (cuda for processing), MPI for communicating best solutions
	for (int i = 0; i < iterations; ++i)
	{
		
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
