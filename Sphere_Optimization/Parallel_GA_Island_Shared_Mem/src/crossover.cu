#include "../include/crossover.cuh"

__device__
void arithmetic_crossover(curandState *d_state, double *islandParent, double *islandPop, 
                          const int p, int tid, const int individualsPerIsland, const float crossoverProbability, const float alpha) {
    float myrand = curand_uniform(&d_state[tid]);

    if (myrand < crossoverProbability) {
        if (threadIdx.x < (individualsPerIsland/2)) {
            for (unsigned int i = 0; i < p; i++) {
                islandPop[threadIdx.x * p + i] = alpha * islandParent[threadIdx.x * p + i] + (1-alpha) * islandParent[(threadIdx.x+(individualsPerIsland/2)) * p + i];
                islandPop[(threadIdx.x+(individualsPerIsland/2)) * p + i] = (1-alpha) * islandParent[threadIdx.x * p + i] + alpha * islandParent[(threadIdx.x+(individualsPerIsland/2)) * p + i];
            }
        } 
    } else {
        for (unsigned int i = 0; i < p; i++) {
            islandPop[threadIdx.x * p + i] = islandParent[threadIdx.x * p + i];
            islandPop[(threadIdx.x+(individualsPerIsland/2)) * p + i] = islandParent[(threadIdx.x+(individualsPerIsland/2)) * p + i];
        }   
    }
}