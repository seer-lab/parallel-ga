#ifndef __crossover_cuh__
#define __crossover_cuh__

#include "utils.h"

__device__
void arithmetic_crossover(curandState *d_state, double *islandParent, double *islandPop, 
                          const int p, int tid, const int individualsPerIsland, const float crossoverProbability, const float alpha);

#endif