#include "../include/migration.cuh"

// Helper function for elitism
// Determines best individual in the population
__device__
unsigned int bestIndividual(double *a, int s, int n) {
    unsigned int i, index = 0;
    double diff = DBL_MAX;
    
    for (i = s; i < n; i++) {
        double absVal = fabsf(a[i]);
        if (absVal < diff) {
            index = i;
            diff = absVal;
        } else if (absVal == diff && a[i] > 0 && a[index] < 0) {
            index = i;
        }
    }

    return index;
}

__device__
void migration(double* population, double *fitness, const int p, int tid, 
               const int elitism, const int individualsPerIsland, const int islands) {
    unsigned int minIndex;

    if (tid % islands < elitism) {
        minIndex = bestIndividual(fitness, tid-threadIdx.x, (tid-threadIdx.x)+individualsPerIsland);
        fitness[minIndex] = DBL_MAX;
    }

    if (tid % islands < elitism) {
        for (unsigned int i = 0; i < p; i++) {
            population[(((blockIdx.x+1)%islands) * blockDim.x + threadIdx.x) * p + i] = population[minIndex * p + i];
        }
    }
}