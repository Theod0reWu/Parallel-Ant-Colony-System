mpi-cpu: main-cpu.c tsp-ant-colony.c
	mpixlc -O3 -c main-cpu.c -o main-cpu.o
	gcc -O3 -c tsp-ant-colony.c -o tsp-ant-colony.o
	mpixlc -O3 main-cpu.o tsp-ant-colony.o -o run-cpu-exe -lstdc++