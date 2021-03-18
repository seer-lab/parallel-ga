#ifndef __evaluation_cuh__
#define __evaluation_cuh__

#include "utils.h"

__device__
double square(double x);

__device__
void evaluation(double *islandFitness, double *islandPop, const int p);

#endif
