# Parallel-Ant-Colony-System
Implementation of a the Ant Colony System optimization, using MPI and CUDA. Made to run on the AiMOS supercomputer.

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





