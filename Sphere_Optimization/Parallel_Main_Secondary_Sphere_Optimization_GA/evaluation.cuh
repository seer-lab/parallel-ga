#ifndef __evaluation_cuh__
#define __evaluation_cuh__

#include "utils.h"

__device__
double square(double x);

__global__
void sphere_eval(double *p, double *f, int numRows, int numCols);

void evaluation(int warp, double* h_population, double* h_fitness, double *d_population, double *d_fitness, const int row, const int col, size_t bytesPopulation, size_t bytesFitness);

#endif
