#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>

int ** SEND_BUF = NULL;
int ** RECV_BUF = NULL;

// external cuda functions
void freeCudaGlobal();
void setup_probelm_tsp(int myrank, int grid_size, int thread_count, double ** nodes, size_t num_coords);

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
	if (argc != 6)
	{
		printf("Incorrect number of provided parameters. %i\n", argc);
		printf("Use -$ <number of colonies> <total ants> <iterations> <thread_count> <input file name> \n");
		return 0;
	}
	unsigned int colonies = 0;
	unsigned int total_ants = 0;
	unsigned int iterations = 0;
	unsigned int thread_count = 0;
	char * filename;

	colonies = atoi(argv[1]);
	total_ants = atoi(argv[2]);
	iterations = atoi(argv[3]);
	thread_count = atoi(argv[4]);
	filename = argv[5];

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

	printf("%i | %lu | (%f, %f)\n", myrank, num_coords, coords[1][0], coords[1][1]);

	// set up colony for this process (create ants, init pheromone trails)
	setup_probelm_tsp(myrank, (total_ants / colonies + thread_count - 1) / thread_count, thread_count, coords,  num_coords);

	// set up MPI reception buffer for processing asynchronous communication
	MPI_Irecv(, worldSize, MPI_CHAR, aboveRank, 'A', MPI_COMM_WORLD, &recv_request_above);

	// execution loop (cuda for processing), MPI for communicating best solutions

	// free memory
	free(coords[0]);
	free(coords[1]);
	free(coords);

	// free cuda memory
	freeCudaGlobal();

	// Finalize the MPI environment.
	MPI_Finalize();
}
