#ifndef __selection_cuh__
#define __selection_cuh__

#include "utils.h"

__device__
unsigned int bestIndividual(double *a, unsigned int *b, int n);

__device__
void selection(curandState *d_state, double *islandParent, double *islandPop, 
               double *islandFitness, const int p, int tid, const int individualsPerIsland);
                
#endif