#ifndef __evaluation_cuh__
#define __evaluation_cuh__

#include "utils.h"
#include "../../constants_serial.h"

__device__
double square(double x);

__global__
void sphere_eval(double *p, double *f, int numRows, int numCols);

__global__
void rastrigin_eval(double *p, double *f, int numRows, int numCols);

__global__
void ackley_eval(double *p, double *f, int numRows, int numCols);

__global__
void griewank_eval(double *p, double *f, int numRows, int numCols);

void evaluation(const int warp1, double* h_population, double* h_fitness, double *d_population, double *d_fitness, 
                const int populationSize, const int p, size_t bytesPopulation, size_t bytesFitness);

#endif
