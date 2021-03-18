#include "../include/migration.cuh"

// Helper function for elitism
// Determines best individual in the population
__device__
int bestIndividual(double *a, int n) {
    unsigned int i, index = 0;
    double diff = DBL_MAX;
    
    for (i = 0; i < n; i++) {
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
void migration(double* population, double *islandPop, double *islandFitness, const int p, int tid, 
               const int elitism, const int individualsPerIsland, const int islands) {
    unsigned int minIndex;
    
    if (threadIdx.x < elitism) {
        minIndex = bestIndividual(islandFitness, individualsPerIsland);
        islandFitness[minIndex] = DBL_MAX;
    }

    for (unsigned int i = 0; i < p; i++)
        population[tid * p + i] = islandPop[threadIdx.x * p + i];
    __syncthreads();

    if (threadIdx.x < elitism) {
        for (unsigned int i = 0; i < p; i++) {
            population[(((blockIdx.x+1)%islands) * blockDim.x + threadIdx.x) * p + i] = islandPop[minIndex * p + i];
        }
    }
}