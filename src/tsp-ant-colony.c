#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// setup to run withou CUDA (unfinished)

// struct idea, but no structs, cause using them in CUDA is a pain
typedef struct {
  double ** nodes;
  double ** edge_weights;
} Problem;


double ** create_adj_matrix(int num_nodes)
{
	double **  m = (double **) calloc(num_nodes, sizeof(double*));
	for (int i = 0; i < num_nodes; ++i)
	{
		m[i] = (double *) calloc(num_nodes,  sizeof(double));
	}
	return m;
}

double ** init_edge_weights_tsp(double ** nodes, int num_nodes)
{	
	double **  weights = create_adj_matrix(num_nodes);

	for (int y = 0; y < num_nodes; ++y)
	{
		for (int x = 0; x < num_nodes; ++x)
		{
			weights[y][x] = 1 / pow(pow(nodes[x][0] - nodes[y][0], 2) + pow(nodes[x][0] - nodes[y][0], 2), .5);
		}
	}
	return weights;
}
