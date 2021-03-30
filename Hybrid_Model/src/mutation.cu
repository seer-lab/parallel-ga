#include "../include/mutation.cuh"

__device__
void mutation(curandState *d_state, double *population, const int p, int tid, 
              const float lowerBound, const float upperBound, const float mutationProbability) {

    for (unsigned int i = 0; i < p; i++) {
        float myrand = curand_uniform(&d_state[tid]);
        if (myrand < mutationProbability) 
            population[tid * p + i] = lowerBound + (curand_uniform(&d_state[tid])) * (upperBound - lowerBound);
    }
}