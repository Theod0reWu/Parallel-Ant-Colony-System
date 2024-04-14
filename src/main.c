#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>

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

	// set up colonies (create ants, init pheromone trails)

	// set up MPI reception buffer for processing asynchronous communication

	// execution loop (cuda for processing), MPI for communicating best solutions



	// Finalize the MPI environment.
	MPI_Finalize();
}
