#include "../include/evaluation.cuh"


__device__
double square(double x) { return x * x; }

__device__
void evaluation(double *fitness, double *population, const int p, int tid) {

    for (unsigned int i = 0; i < p; i++)
        fitness[tid] += square(population[tid * p + i]);
}