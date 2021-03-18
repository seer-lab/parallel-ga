#include "../include/crossover.cuh"

__device__
void arithmetic_crossover(curandState *d_state, double *parents, double *population, 
                          const int p, int tid, const int individualsPerIsland, const float crossoverProbability, const float alpha) {
    float myrand = curand_uniform(&d_state[tid]);
    int threshold = (((individualsPerIsland*blockIdx.x+individualsPerIsland)/2)+((individualsPerIsland/2)*blockIdx.x));
    if (myrand < crossoverProbability) {
        if (tid < threshold) {
            for (unsigned int i = 0; i < p; i++) {
                population[tid * p + i] = alpha * parents[tid * p + i] + (1-alpha) * parents[(tid+individualsPerIsland/2) * p + i];
                population[(tid+individualsPerIsland/2) * p + i] = (1-alpha) * parents[tid * p + i] + alpha * parents[(tid+individualsPerIsland/2) * p + i];
            }
        } 
    } else {
        for (unsigned int i = 0; i < p; i++) {
            population[tid * p + i] = parents[tid * p + i];
            population[(tid+individualsPerIsland/2) * p + i] = parents[(tid+individualsPerIsland/2) * p + i];
        }   
    }
}