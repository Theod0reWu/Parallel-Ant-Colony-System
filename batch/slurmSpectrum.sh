#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=4

module load xl_r spectrum-mpi cuda/11.2

#mpirun --bind-to core --report-bindings -np 8 ./mpi-cuda-exe
mpirun -np 8 ./mpi-cuda-exe 5 32 65 8
mpirun -np 8 ./mpi-cuda-exe 5 32 66 8
mpirun -np 8 ./mpi-cuda-exe 5 32 67 8
mpirun -np 8 ./mpi-cuda-exe 5 32 68 8
~                                      