mpi-cuda: main.c colony-run.cu
	mpixlc -O3 main.c -c -o mpi-cuda-xlc.o
	nvcc -O3 -arch=sm_70 colony-run.cu -c -o mpi-cuda-nvcc.o
	mpixlc -O3 mpi-cuda-xlc.o mpi-cuda-nvcc.o -o ../batch/run-exe -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++