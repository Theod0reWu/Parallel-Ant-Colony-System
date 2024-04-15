#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>

double * create_colonies(int num_colonies, int ants_per_colony)
{	
	
	return 0;
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
		printf("Incorrect number of provided parameters. use -$ <number of colonies> <total ants> <iterations> <thread_count> \n");
		return 0;
	}
	unsigned int colonies = 0;
	unsigned int total_ants = 0;
	unsigned int iterations = 0;
	unsigned int thread_count = 0;

	colonies = atoi(argv[1]);
	total_ants = atoi(argv[2]);
	iterations = atoi(argv[3]);
	thread_count = atoi(argv[4]);

	// set up colonies (create ants, init pheromone trails)

	// set up MPI reception buffer for processing asynchronous communication

	// execution loop (cuda for processing), MPI for communicating best solutions



	// Finalize the MPI environment.
	MPI_Finalize();
}
