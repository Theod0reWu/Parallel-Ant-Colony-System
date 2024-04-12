#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>

int main(int argc, char** argv) {

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

	// Finalize the MPI environment.
	MPI_Finalize();
}
