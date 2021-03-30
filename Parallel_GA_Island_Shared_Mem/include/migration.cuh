#ifndef __migration_cuh__
#define __migration_cuh__

#include "utils.h"

__device__
int bestIndividual(double *a, int n);

__device__
void migration(double* population, double *islandPop, double *islandFitness, const int p, int tid, 
               const int elitism, const int individualsPerIsland, const int islands);

#endif

