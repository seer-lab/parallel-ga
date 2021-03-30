#ifndef __evaluation_cuh__
#define __evaluation_cuh__

#include "utils.h"

__device__
double square(double x);

__device__
void sphere(double *fitness, double *population, const int p, int tid);

__device__
void rastrigin(double *fitness, double *population, const int p, int tid);

__device__
void ackley(double *fitness, double *population, const int p, int tid);

__device__
void griewank(double *fitness, double *population, const int p, int tid);

#endif
