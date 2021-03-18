#include "../include/evaluation.cuh"


__device__
double square(double x) { return x * x; }

__device__
void evaluation(double *islandFitness, double *islandPop, const int p) {
    for (unsigned int i = 0; i < p; i++)
        islandFitness[threadIdx.x] += square(islandPop[threadIdx.x * p + i]);
}