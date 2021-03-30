#ifndef __evaluation_cuh__
#define __evaluation_cuh__

#include "utils.h"

__device__
double square(double x);

__device__
void sphere(double *islandFitness, double *islandPop, const int p);

__device__
void rastrigin(double *islandFitness, double *islandPop, const int p);

__device__
void ackley(double *islandFitness, double *islandPop, const int p);

__device__
void griewank(double *islandFitness, double *islandPop, const int p);

#endif
