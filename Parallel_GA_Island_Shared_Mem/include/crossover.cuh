#ifndef __crossover_cuh__
#define __crossover_cuh__

#include "utils.h"

__device__
void arithmetic_crossover(curandState *d_state, double *islandParent, double *islandPop, 
                          const int p, int tid, const int individualsPerIsland, const float crossoverProbability, const float alpha);

__device__
double calc_B(float u, int nc);

__device__
void simulated_binary_crossover(curandState *d_state, double *islandParent, double *islandPop, 
                        const int p, int tid, const int individualsPerIsland, const float crossoverProbability, const int nc);

__device__
void line_crossover(curandState *d_state, double *islandParent, double *islandPop, 
                    const int p, int tid, const int individualsPerIsland, const float crossoverProbability);                        

#endif