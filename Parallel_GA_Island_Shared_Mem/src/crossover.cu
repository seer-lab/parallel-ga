#include "../include/crossover.cuh"

#define d 0.25

__device__
void arithmetic_crossover(curandState *d_state, double *islandParent, double *islandPop, 
                          const int p, int tid, const int individualsPerIsland, const float crossoverProbability, const float alpha) {
    float myrand = curand_uniform(&d_state[tid]);
                
    if (myrand < crossoverProbability) {
        if (threadIdx.x < (individualsPerIsland/2)) {
            for (unsigned int i = 0; i < p; i++) {
                islandPop[threadIdx.x * p + i] = alpha * islandParent[threadIdx.x * p + i] + (1-alpha) * 
                                                islandParent[(threadIdx.x+(individualsPerIsland/2)) * p + i];

                islandPop[(threadIdx.x+(individualsPerIsland/2)) * p + i] = (1-alpha) * islandParent[threadIdx.x * p + i] + alpha * 
                                                islandParent[(threadIdx.x+(individualsPerIsland/2)) * p + i];
            }
        } 
    } else {
        for (unsigned int i = 0; i < p; i++) {
            islandPop[threadIdx.x * p + i] = islandParent[threadIdx.x * p + i];
        }   
    }
}

__device__
double calc_B(float u, int nc) {
    if (u <= 0.5f) 
        return pow(2.0*u, 1.0/(nc+1));
    else 
        return pow(1.0/(2.0*(1.0-u)), 1.0/(nc+1));
}

__device__
void simulated_binary_crossover(curandState *d_state, double *islandParent, double *islandPop, 
                                const int p, int tid, const int individualsPerIsland, const float crossoverProbability, const int nc) {
    float myrand = curand_uniform(&d_state[tid]);
    
    if (threadIdx.x < (individualsPerIsland/2)) {                        
        if (myrand < crossoverProbability) {
            for (unsigned int i = 0; i < p; i++) {
                float u = curand_uniform(&d_state[tid]);
                double b = calc_B(u, nc);
                
                islandPop[threadIdx.x * p + i] = 0.5*(((1+b) * islandParent[threadIdx.x * p + i]) + 
                            ((1-b) * islandParent[(threadIdx.x+(individualsPerIsland/2)) * p + i]));

                islandPop[(threadIdx.x+(individualsPerIsland/2)) * p + i] = 0.5*(((1-b) * islandParent[threadIdx.x * p + i]) + 
                            ((1+b) * islandParent[(threadIdx.x+(individualsPerIsland/2)) * p + i]));
            }
        } else {
            for (unsigned int i = 0; i < p; i++) {
                islandPop[threadIdx.x * p + i] = islandParent[threadIdx.x * p + i];
                islandPop[(threadIdx.x+(individualsPerIsland/2)) * p + i] = islandParent[(threadIdx.x+(individualsPerIsland/2)) * p + i];
            }   
        }
    } 
}

__device__
void line_crossover(curandState *d_state, double *islandParent, double *islandPop, 
                    const int p, int tid, const int individualsPerIsland, const float crossoverProbability) {
    float myrand = curand_uniform(&d_state[tid]);
    
    if (threadIdx.x < (individualsPerIsland/2)) {                        
        if (myrand < crossoverProbability) {
            for (unsigned int i = 0; i < p; i++) {
                float alpha = -d + curand_uniform(&d_state[tid]) * ((1+d) - -d);
                float alpha2 = -d + curand_uniform(&d_state[tid]) * ((1+d) - -d);
                
                islandPop[threadIdx.x * p + i] = ((islandParent[threadIdx.x * p + i]) + 
                                alpha * (islandParent[(threadIdx.x+(individualsPerIsland/2)) * p + i] - islandParent[threadIdx.x * p + i]));

                islandPop[(threadIdx.x+(individualsPerIsland/2)) * p + i] = ((islandParent[(threadIdx.x+(individualsPerIsland/2)) * p + i]) + 
                                alpha2 * (islandParent[threadIdx.x * p + i] - islandParent[(threadIdx.x+(individualsPerIsland/2)) * p + i]));
            }
        } else {
            for (unsigned int i = 0; i < p; i++) {
                islandPop[threadIdx.x * p + i] = islandParent[threadIdx.x * p + i];
                islandPop[(threadIdx.x+(individualsPerIsland/2)) * p + i] = islandParent[(threadIdx.x+(individualsPerIsland/2)) * p + i];
            }   
        }
    } 
}