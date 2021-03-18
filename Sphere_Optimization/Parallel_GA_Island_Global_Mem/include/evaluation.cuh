#ifndef __evaluation_cuh__
#define __evaluation_cuh__

#include "utils.h"

__device__
double square(double x);

__device__
void evaluation(double *fitness, double *population, const int p, int tid);

#endif
