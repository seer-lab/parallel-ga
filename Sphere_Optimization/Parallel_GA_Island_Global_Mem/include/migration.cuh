#ifndef __migration_cuh__
#define __migration_cuh__

#include "utils.h"

__device__
unsigned int bestIndividual(double *a, int s, int n);

__device__
void migration(double* population, double *fitness, const int p, int tid, 
               const int elitism, const int individualsPerIsland, const int islands);

#endif

