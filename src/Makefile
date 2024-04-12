mpi-cuda: mpi-cuda.c mpi-cuda.cu
	mpixlc -O3 mpi-cuda.c -c -o mpi-cuda-xlc.o
	nvcc -O3 -arch=sm_70 mpi-cuda.cu -c -o mpi-cuda-nvcc.o
	mpixlc -O3 mpi-cuda-xlc.o mpi-cuda-nvcc.o -o mpi-cuda-exe -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++