#include "../include/crossover.cuh"

#define d 0.25

__device__
void arithmetic_crossover(curandState *d_state, double *parents, double *population, 
                          const int p, int tid, const int individualsPerIsland, const float crossoverProbability, const float alpha) {
    float myrand = curand_uniform(&d_state[tid]);
    int threshold = (((individualsPerIsland*blockIdx.x+individualsPerIsland)/2)+((individualsPerIsland/2)*blockIdx.x));
    
    if (tid < threshold) {
        if (myrand < crossoverProbability) {
            for (unsigned int i = 0; i < p; i++) {
                population[tid * p + i] = alpha * parents[tid * p + i] + (1-alpha) * parents[(tid+individualsPerIsland/2) * p + i];

                population[(tid+individualsPerIsland/2) * p + i] = (1-alpha) * parents[tid * p + i] + 
                                                                    alpha * parents[(tid+individualsPerIsland/2) * p + i];
            }
        } else {
            for (unsigned int i = 0; i < p; i++) {
                population[tid * p + i] = parents[tid * p + i];
                population[(tid+individualsPerIsland/2) * p + i] = parents[(tid+individualsPerIsland/2) * p + i];
            }   
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
void simulated_binary_crossover(curandState *d_state, double *parents, double *population, 
                                const int p, int tid, const int individualsPerIsland, const float crossoverProbability, const int nc) {
    float myrand = curand_uniform(&d_state[tid]);
    int threshold = (((individualsPerIsland*blockIdx.x+individualsPerIsland)/2)+((individualsPerIsland/2)*blockIdx.x));
    
    if (tid < threshold) {
        if (myrand < crossoverProbability) {
            for (unsigned int i = 0; i < p; i++) {
                float u = curand_uniform(&d_state[tid]);
                double b = calc_B(u, nc);
                population[tid * p + i] = 0.5*(((1+b) * parents[tid * p + i]) + ((1-b) * parents[(tid+individualsPerIsland/2) * p + i]));

                population[(tid+individualsPerIsland/2) * p + i] = 0.5*(((1-b) * parents[tid * p + i]) + 
                                                                    ((1+b) * parents[(tid+individualsPerIsland/2) * p + i]));
            }
        } else {
            for (unsigned int i = 0; i < p; i++) {
                population[tid * p + i] = parents[tid * p + i];
                population[(tid+individualsPerIsland/2) * p + i] = parents[(tid+individualsPerIsland/2) * p + i];
            }   
        }
    } 
}

__device__
void line_crossover(curandState *d_state, double *parents, double *population, 
                                const int p, int tid, const int individualsPerIsland, const float crossoverProbability) {
    float myrand = curand_uniform(&d_state[tid]);
    int threshold = (((individualsPerIsland*blockIdx.x+individualsPerIsland)/2)+((individualsPerIsland/2)*blockIdx.x));
    
    if (tid < threshold) {
        if (myrand < crossoverProbability) {
            for (unsigned int i = 0; i < p; i++) {
                float alpha = -d + curand_uniform(&d_state[tid]) * ((1+d) - -d);
                float alpha2 = -d + curand_uniform(&d_state[tid]) * ((1+d) - -d);

                population[tid * p + i] = (parents[tid * p + i]) + alpha * 
                                        (parents[(tid+individualsPerIsland/2) * p + i] - parents[tid * p + i]);

                population[(tid+individualsPerIsland/2) * p + i] = (parents[(tid+individualsPerIsland/2) * p + i]) + alpha2 * 
                                        (parents[tid * p + i] - parents[(tid+individualsPerIsland/2) * p + i]);
            }
        } else {
            for (unsigned int i = 0; i < p; i++) {
                population[tid * p + i] = parents[tid * p + i];
                population[(tid+individualsPerIsland/2) * p + i] = parents[(tid+individualsPerIsland/2) * p + i];
            }   
        }
    } 
}