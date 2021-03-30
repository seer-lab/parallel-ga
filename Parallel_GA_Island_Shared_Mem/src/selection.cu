#include "../include/selection.cuh"
#define tournamentSize 6

__device__
unsigned int bestIndividual(double *a, unsigned int *b, int n) {
    
    unsigned int i, index = b[0];
    double diff = DBL_MAX;

    for (i = 0; i < n; i++) {
        double absVal = fabsf(a[b[i]]);
        if (absVal < diff) {
            index = b[i];
            diff = absVal;
        } else if (absVal == diff && a[b[i]] > 0 && a[index] < 0) {
            index = b[i];
        }
    }

    return index;
}

__device__
void selection(curandState *d_state, double *islandParent, double *islandPop, 
               double *islandFitness, const int p, int tid, const int individualsPerIsland) {
                   
    // Individuals selected for tournament
    unsigned int tournamentPool[tournamentSize];

    for (unsigned int i = 0; i < tournamentSize; i++) {
        float myrand = curand_uniform(&d_state[tid]);
        myrand *= (individualsPerIsland-1+0.999999);
        int value = (int)truncf(myrand);

        tournamentPool[i] = value;
    }

    // select best individual in tournament
    unsigned int parentIndex = bestIndividual(islandFitness, tournamentPool, tournamentSize);
    
    // Copying best individual from tournament to island parent 
    for (unsigned int i = 0; i < p; i++)
        islandParent[threadIdx.x * p + i] = islandPop[parentIndex * p + i];

}