#ifndef __mutation_cuh__
#define __mutation_cuh__

#include "utils.h"

__device__
void mutation(curandState *d_state, double *temp_islandPop, const int p, int tid, 
              const float lowerBound, const float upperBound, const float mutationProbability);

#endif