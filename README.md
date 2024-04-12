# Parallel-Ant-Colony-System
Implementation of a the Ant Colony System optimization, using MPI and CUDA. Made to run on the AiMOS supercomputer.

## Background
The ant colony optimization algorithm base unit of computation is called an ant. Like real world ants, an ant in the algorithm will explore a world, looking for rewards. In the algorithm the ant will look for a solution to an optimization problem. To apply an ant colony algorithm, the optimization problem needs to be converted into the problem of finding the shortest path on a weighted graph.
<br>
The procedure ACO_MetaHeuristic is:
```
  while not terminated do  
        generateSolutions()  
        daemonActions()  
        pheromoneUpdate()  
    repeat  
  end procedure  
```
Each ant needs to construct a solution to move through the graph. To select the next edge in its tour, an ant will consider the length of each edge available from its current position, as well as the corresponding pheromone level.In general, ants move between states with probability:
![image](https://github.com/Theod0reWu/Parallel-Ant-Colony-System/assets/43049406/aa3709ff-2e19-44ab-a655-9abf602e40a1)
