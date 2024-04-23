# Parallel-Ant-Colony-System
Implementation of the Ant Colony System optimization algorithm, using MPI and CUDA. Made to run on the AiMOS supercomputer.

## Background
The ant colony optimization algorithm base unit of computation is called an ant. Like real world ants, an ant in the algorithm will explore a world (randomly), looking for rewards. In the algorithm the ant will look for a solution to an optimization problem. To apply an ant colony algorithm, the optimization problem needs to be converted into the problem of finding the shortest path on a weighted graph. Ants will leave pheromones to influence the exploration of other ants. The evaluation of the solution will decide how pheromones are preserved.
<br>
The highlevel pseudocode is:
```
  while not terminated do  
        generateSolutions()  
        daemonActions()  
        pheromoneUpdate()  
    repeat  
  end procedure  
```
The generateSolutions() function will mainly take advantage of some heuristic function to generate a path. The dameonActions() function will mainly compare the paths found by different “ants”. Finally the pheromoneUpdate() function will update the pheromone information that is shared across the “ants”. 

Each ant needs to construct a solution to move through the graph. To select the next edge in its tour, an ant will consider the length of each edge available from its current position, as well as the corresponding pheromone level. In general, ants move between states with probability:
<br>
![image](https://github.com/Theod0reWu/Parallel-Ant-Colony-System/assets/43049406/aa3709ff-2e19-44ab-a655-9abf602e40a1)

![image](https://github.com/Theod0reWu/Parallel-Ant-Colony-System/assets/43049406/5ae88e0d-6d57-4b79-b203-b34c99497057) is the amount of pheromone deposited for transition from state x to y. ![image](https://github.com/Theod0reWu/Parallel-Ant-Colony-System/assets/43049406/00a17e74-1f56-4431-a07c-1129ddaa0a7c)  is the desirability of state transition xy.

Trails are usually updated when all ants have completed their solution, increasing or decreasing the level of trails corresponding to moves that were part of "good" or "bad" solutions, respectively. An example of a global pheromone updating rule is
<br>
![image](https://github.com/Theod0reWu/Parallel-Ant-Colony-System/assets/43049406/d66ba134-3fb8-4943-99c6-ee5a73c43174) 
<br>
![image](https://github.com/Theod0reWu/Parallel-Ant-Colony-System/assets/43049406/7ae78bf3-9d1d-445d-90a6-11ef07d9a0df)
<br>
(Source: Wikipedia)

## Input file format
Each line of the input file should contain a coordinate in the form:
```
<float>,<float>
```
This represents a node in the TSP and distances between nodes are calculated as the euclidean distance. <br>
In data, generate_data.py can be used to create an input file:
```
-$ generate_data.py <file name> <number of points> <disc or circle>
```

## Implementation details

The code is written in C with MPI and CUDA. Each MPI rank will host its own colony consisting of an equal number of ants. Each iterations is run with CUDA to process its ants.

## Compilation and Execution

Using the makefile to compile. You need to have spectrum MPI and at least cuda 11.0 
<br>
Compile with:
```
-$ make
```
This will generate an executable in the batch folder. To run see below or use the batch file:
```
-$ mpirun -np <mpi ranks> ../batch/run-exe <number of ants> <iterations> <threads per block> <path to input data>
```

## Results
This path was generated created with 256 ants in 2 colonies at 10 iterations:
![image](https://github.com/Theod0reWu/Parallel-Ant-Colony-System/assets/43049406/bfcd0dcf-17ce-4141-86ad-a84ec943d9d3)

More intensive training yielded promising results. The image below shows the results of 512 ants in 8 colonies for 10 iterations. There were 1000 points arrange in a circle:
![image](https://github.com/Theod0reWu/Parallel-Ant-Colony-System/assets/43049406/b89c98f7-9fe7-4d31-a967-5ab9cfb512e8)

