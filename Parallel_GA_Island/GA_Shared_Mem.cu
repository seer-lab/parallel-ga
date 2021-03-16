#include <stdio.h> 
#include <stdlib.h>
#include <cmath> 
#include <cstring>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <cfloat>
#include <math.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <curand.h>
#include <curand_kernel.h>

using std::vector;
using std::cout;
using std::endl;

// Island Parameters
#define islands 64
#define individualsPerIsland 2048/islands

#define SHMEN_SIZE (10 * individualsPerIsland * 8)

#define elitism 1
#define tournamentSize 6
#define crossoverProbability 0.9f
#define mutationProbability 0.05f
#define alpha 0.5f
#define numGen 10000

__global__
void setup_kernel (curandState* state, unsigned long seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init (seed, tid , 0, &state[tid]);
}

__device__
double square(double x) { return x * x; }

__device__
void evaluation(double *islandFitness, double *islandPop, const int p) {
    for (unsigned int i = 0; i < p; i++)
        islandFitness[threadIdx.x] += square(islandPop[threadIdx.x * p + i]);
}


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
void selection(curandState *d_state, double *islandParent, double *islandPop, double *islandFitness, const int p, int tid) {
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

// Could be an error here
__device__
void crossover(curandState *d_state, double *islandParent, double *islandPop, const int p, int tid) {
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

__device__
void mutation(curandState *d_state, double *temp_islandPop, const int p, int tid) {
    for (unsigned int i = 0; i < p; i++) {
        float myrand = curand_uniform(&d_state[tid]);
        if (myrand < mutationProbability) 
        temp_islandPop[threadIdx.x * p + i] = -5.12 + (curand_uniform(&d_state[tid])) * (5.12 - -5.12);
    }
}

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
void migration(double* population, double *islandPop, double *islandFitness, const int p, int tid) {
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

__global__
void gpu_GA(curandState *d_state, double* population, double* fitness, const int populationSize, const int p, int numGenerations) {

    // Allocating shared memory for each island
    __shared__ double islandPop[SHMEN_SIZE];
    __shared__ double islandFitness[individualsPerIsland * 8];
    __shared__ double islandParent[SHMEN_SIZE];

    // Thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Checking for migration
    bool migrationFlag = false;

    // Load elements into shared memory
    for (unsigned int i = 0; i < p; i++)
        islandPop[threadIdx.x * p + i] = population[tid * p + i];
    __syncthreads();

    // On first generation evaluate current population
    // Else copy over previous fitness scores to respective islands
    if (numGenerations == numGen)
        evaluation(islandFitness, islandPop, p);
    else
        islandFitness[threadIdx.x] = fitness[tid];
    __syncthreads();

    // Every 5 generations migrate individuals to n+1 islands
    if (numGenerations % 5 == 0)
        migrationFlag = true;

    // Tournament Selection
    selection(d_state, islandParent, islandPop, islandFitness, p, tid);
    __syncthreads();
    
    // Arithmetic Crossover 
    crossover(d_state, islandParent, islandPop, p, tid);
    __syncthreads();

    // TODO: Guassian Mutation (Look into implementing guassian mutation)
    mutation(d_state, islandPop, p, tid);
    __syncthreads();

    if (migrationFlag)
        migration(population, islandPop, islandFitness, p, tid);
    __syncthreads();

    islandFitness[threadIdx.x] = 0;
    __syncthreads();

    // Evaluation for each individual
    evaluation(islandFitness, islandPop, p);
    __syncthreads();
    
    // Copy back to global memory
    if (!migrationFlag) {
        for (unsigned int i = 0; i < p; i++)
            population[tid * p + i] = islandPop[threadIdx.x * p + i];
    }

    fitness[tid] = islandFitness[threadIdx.x];
}   

void initPopulation(double* population, std::vector<float> bounds, const int row, const int col) {

    for (unsigned int i = 0; i < row; i++)
        for (unsigned int j = 0; j < col; j++)
            *(population + i*col + j) = bounds[0] + ((double)rand() / RAND_MAX) * (bounds[1] - bounds[0]);

}

void parallelGA(double* h_population, 
                double* h_fitness, 
                const int populationSize, 
                const int p, 
                size_t bytesPopulation, 
                size_t bytesFitness,
                int numGenerations,
                curandState *&d_state) {
    
    // Allocating device memory
    double *d_population, *d_fitness;

    cudaMalloc(&d_population, bytesPopulation);
    cudaMalloc(&d_fitness, bytesFitness);
    
    // Copying population to device (intend to remove in the future)
    cudaMemcpy(d_population, h_population, bytesPopulation, cudaMemcpyHostToDevice);
    
    while (numGenerations > 0) {
        gpu_GA<<<islands, 
                 individualsPerIsland,
                 (SHMEN_SIZE*2) + (individualsPerIsland * 8)>>>
                 (d_state, d_population, d_fitness, populationSize, p, numGenerations);
        cudaDeviceSynchronize(); 
        numGenerations--;
    }

    // Copy population and fitness back to host
    cudaMemcpy(h_population, d_population, bytesPopulation, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fitness, d_fitness, bytesFitness, cudaMemcpyDeviceToHost);

    cudaFree(d_population);
    cudaFree(d_fitness);
    cudaFree(d_state);

}

void print2dvec(double *v, int r, int c) {
    for (unsigned int i = 0; i < r; i++) {
        cout << "[ ";
        for (unsigned int j = 0; j < c; j++) 
            cout << v[i * c + j] << " ";
        cout << "]\n";
    }
}

void printvec(double *v, int n) {
    for (unsigned int i = 0; i < n; i++)
		cout << v[i] << "\n";
}


int main() {

    // Optimization parameters for Sphere function
    vector<float> bounds{-5.12, 5.12};

    // GA parameters
    const int p = 10; // # of genes per individual
    const int populationSize = 2048; 
    int numGenerations = 10000; 

    // Intialization for random number generator
    time_t t;
    srand((unsigned) time(&t));

    // Vector size
    size_t bytesPopulation = p * populationSize * sizeof(double);
    size_t bytesFitness = populationSize * sizeof(double);

    // Initilize vectors
    double *population, *fitness;

    // Allocate memory
    population = (double*)malloc(bytesPopulation);
    fitness = (double*)malloc(bytesFitness);

    // Initialize Population 
    initPopulation(population, bounds, populationSize, p);

    // cuRand setup
    curandState *d_state;
    cudaMalloc(&d_state, islands*individualsPerIsland*sizeof( curandState ) );
    setup_kernel<<<islands,individualsPerIsland>>>(d_state, (unsigned long) t );

    // GA
    parallelGA(population, fitness, populationSize, p, bytesPopulation, bytesFitness, numGenerations, d_state);

    printvec(fitness, populationSize);

    return 0;
}