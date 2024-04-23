#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<string.h>
#include <time.h>

#include<math.h>
#include<limits.h>

// setup to run withou CUDA (unfinished)

// struct idea, but no structs, cause using them in CUDA is a pain
typedef struct {
  double ** nodes;
  double ** edge_weights;
} Problem;

double ALPHA = 1;
double BETA = 1;

// adjacency matrices
double ** EDGE_WEIGHTS;
double ** PHER_TRAILS;

//scores of device
size_t ** VISITED;
double * SCORES;

// external for MPI message passing
extern size_t * SEND_BUF;
extern bool SEND_READY;
extern size_t * RECV_BUF;

double BEST_SCORE = -1;

size_t NUM_NODES;
size_t NUM_ANTS;

double INIT_SMALL = 0; //.000001;
double DECAY_RATE = .1;

void displayAdjMatrix(double ** matrix);

double randomDouble()
{
    return ((double)(rand()) / (double)RAND_MAX);
}

double ** createAdjMatrix(size_t num_nodes)
{
    double **  m;
    m = malloc(num_nodes * sizeof(double*));
    for (int i = 0; i < num_nodes; ++i)
    {
        m[i] = malloc(num_nodes * sizeof(double));
    }
    return m;
}

double ** createEdgeWeightsTSP(double ** nodes, int num_nodes)
{   
    double **  weights = createAdjMatrix(num_nodes);

    for (int y = 0; y < num_nodes; ++y)
    {
        for (int x = 0; x < num_nodes; ++x)
        {
            weights[y][x] = 1 / pow(pow(nodes[0][x] - nodes[0][y], 2) + pow(nodes[1][x] - nodes[1][y], 2), .5);
        }
    }
    return weights;
}

// sets up the devices and runs
void setupProblemTSP(int myrank, double ** nodes, size_t num_coords, size_t num_ants)
{
    // create adj matrix for edge weights and phermone trails
    EDGE_WEIGHTS = createEdgeWeightsTSP(nodes, num_coords);
    PHER_TRAILS = createAdjMatrix(num_coords);
    for (int i = 0; i < num_coords; ++i)
    {
        for (int e = 0; e < num_coords; ++e)
        {
            PHER_TRAILS[i][e] = 0;
        }
    }

    srand(time(0));

    // create visited array
    // visited[i] is the path of ant i
    VISITED = malloc(num_ants * sizeof(size_t *));
    for (int i = 0; i < num_ants; i++)
    {
        VISITED[i] = malloc(num_coords * sizeof(size_t *));
    }
    SCORES = malloc(num_ants * sizeof(double));

    NUM_NODES = num_coords;
    NUM_ANTS = num_ants;
}

bool elementOf(size_t * visited, size_t size, size_t at)
{
    for (int i = 0; i < size; ++i)
    {
        if (visited[i] == at)
        {
            return true;
        }
    }
    return false;
}

// run an ant, who creates one solution to the TSP per ant.
void colonyTSP(
    double ** pheromone_trails,
    double ** edge_weights,
    size_t ** visited, 
    double * scores,
    size_t num_ants, size_t num_nodes)
{

    for (size_t index = 0; index < num_ants; index++)
    {
        double total = 0;
        scores[index] = 0;

        // determine what node is next based on probability
        for (size_t step = 0; step < num_nodes; step++)
        {
            if (step == 0)
            {
                // starting city is randomly assigned
                int rand_index = (int) (randomDouble() * ((num_nodes - 1) + 0.99));
                // printf("rand: %i\n",rand_index);
                visited[index][0] = rand_index;
            } else {
                // get the total score of all possible options
                total = 0;
                for (size_t node = 0; node < num_nodes; node++)
                {
                    if (!elementOf(visited[index], step, node))
                    {
                        // printf("step-1: %i node: %i prev: %i index: %i\n", step-1, node, visited[index][step - 1], index);
                        // printf("%f %f\n", pheromone_trails[visited[index][step - 1]][node], edge_weights[visited[index][step - 1]][node]);
                        total += pheromone_trails[visited[index][step - 1]][node] * edge_weights[visited[index][step - 1]][node];
                    }
                }

                double rand = randomDouble() * total;
                double running_sum = 0;
                for (size_t node = 0; node < num_nodes; node++)
                {
                    if (!elementOf(visited[index], step, node))
                    {
                        running_sum += pheromone_trails[visited[index][step - 1]][node] * edge_weights[visited[index][step - 1]][node];
                        if (running_sum >= rand)
                        {
                            visited[index][step] = node;
                            break; // found node stop looking
                        }
                    }
                }
                // since the edge weights are 1 / distance, score is the sum of distances.
                scores[index] += 1 / edge_weights[visited[index][step-1]][visited[index][step]];
            }
        }
        // add the distance from the last to first node
        scores[index] += 1 / edge_weights[visited[index][num_nodes-1]][visited[index][0]];
    }
}


void decayPheromones(double rho, size_t num_nodes)
{
    // decay existing pheromones
    for (int y = 0; y < num_nodes; y++)
    {
        for (int x = 0; x < num_nodes; x++)
        {
            PHER_TRAILS[y][x] *= (1 - rho);
        }
    }
}

// updates the pheromones based on the ant system (Dorigo et al. 1991, Dorigo 1992, Dorigo et al. 1996)
// every ant updates the phermones based on their score
void updatePheromoneTrailsAS(
    double ** pheromone_trails,
    size_t ** visited, 
    double * scores,
    size_t num_ants, size_t num_nodes)
{

    for (size_t index = 0; index < num_ants; index++)
    {
        for (size_t step = 0; step < num_nodes; step++)
        {   
            pheromone_trails[visited[index][(num_nodes + step - 1) % num_nodes]][visited[index][step]] += 1 / scores[index];
        }
    }
}

double getScore(size_t * path, size_t num_nodes)
{
    double score = 0;
    for (size_t step = 0; step < num_nodes; step++)
    {
        score += 1 / EDGE_WEIGHTS[path[(step-1 + num_nodes) % num_nodes]][path[step]];
    }
    return score;
}

//ant colony system (ACS), introduced by Dorigo and Gambardella (1997)
// only updates based on the best ant
void updatePheromoneTrailsACS(
    double ** pheromone_trails, 
    size_t * path, size_t num_nodes,
    double score
    )
{
    for (size_t step = 0; step < num_nodes; step++)
    {   
        pheromone_trails[path[(num_nodes + step - 1) % num_nodes]][path[step]] += 1 / score;
    }
}

// returns the index of the ant with the best score
size_t getBestAnt(size_t num_ants)
{
    double best = SCORES[0];
    size_t idx = 0;
    for (size_t i = 1; i < num_ants; i++)
    {
        if (SCORES[i] < best)
        {
            best = SCORES[i];
            idx = i;
        }
    }
    return idx;
}

void displayAdjMatrix(double ** matrix) {
    for (int y = 0; y < NUM_NODES; ++y)
    {
        for (int x = 0; x < NUM_NODES; x++){
            printf("%lf ", matrix[y][x]);
        }
        printf("\n");
    }
}

void freeAdjMatrix(double ** matrix) {
    for (int y = 0; y < NUM_NODES; ++y)
    {
        free(matrix[y]);
    }
    free(matrix);
}

void updatePheromones(int num_nodes, char * update_rule, bool decay, double rho)
{
    if (decay)
    {
        decayPheromones(rho, NUM_NODES);
    }
    if (strcmp(update_rule, "AS") == 0)
    {
        updatePheromoneTrailsAS(PHER_TRAILS, VISITED, SCORES, NUM_ANTS, NUM_NODES);
    }
    else if (strcmp(update_rule, "ACS") == 0)
    {
        // go look for the best path then update based on that
        updatePheromoneTrailsACS(PHER_TRAILS, SEND_BUF, NUM_NODES, 1);
    }
    else if (strcmp(update_rule, "MESSAGE") == 0)
    {   
        double path_score = getScore(RECV_BUF, NUM_NODES);
        if (path_score > BEST_SCORE)
        {
            updatePheromoneTrailsACS(PHER_TRAILS, RECV_BUF, NUM_NODES, path_score);
        }
    }
}

void colonyRun(size_t num_nodes, size_t num_ants)
{
    colonyTSP(PHER_TRAILS, EDGE_WEIGHTS, VISITED, SCORES, num_ants, NUM_NODES);

    // Calculate the best score and send if better than the best found
    size_t best_idx = getBestAnt(num_ants);
    double best = SCORES[best_idx];
    if (BEST_SCORE == -1 || best < BEST_SCORE)
    {
        printf("Best score: %lf \n", best);
        BEST_SCORE = best;

        // load the best route into send buffer, ready to send
        memcpy(SEND_BUF, VISITED[best_idx], sizeof(size_t) * NUM_NODES);
        SEND_READY = true;
    }
}

void freeGlobal(int num_ants){
    freeAdjMatrix(EDGE_WEIGHTS);
    freeAdjMatrix(PHER_TRAILS);
    for (int i = 0; i < num_ants; i++)
    {
        free(VISITED[i]);
    }
    free(VISITED);
    free(SCORES);
}



