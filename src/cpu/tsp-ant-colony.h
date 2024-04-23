#ifndef TEST_H_INCLUDED
#define TEST_H_INCLUDED

void setupProblemTSP(int myrank, double ** nodes, size_t num_coords, size_t num_ants);
void updatePheromones(int num_nodes, char * update_rule, bool decay, double rho);
void colonyRun(size_t num_nodes, size_t num_ants);
void freeGlobal(int num_ants);

#endif