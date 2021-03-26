#include <stdio.h> 
#include <stdlib.h>
#include <cstring>
#include <cstdlib>
#include <math.h>
#include <algorithm>

#include "include/utils.h" // For cuda imports

#include "include/migration.cuh"
#include "include/mutation.cuh"
#include "include/crossover.cuh"
#include "include/selection.cuh"
#include "include/evaluation.cuh"
#include "include/population.h"

using std::cout;
using std::endl;

// Island Parameters
#define islands 64
#define individualsPerIsland 2048/islands

#define SHMEN_SIZE (10 * individualsPerIsland * 8)

#define elitism 2
#define tournamentSize 6
#define crossoverProbability 0.9f
#define mutationProbability 0.05f
#define alpha 0.25f
#define numGen 100
#define lowerBound -5.12
#define upperBound 5.12

// Initializing CUDA rand
__global__
void setup_kernel (curandState* state, unsigned long seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init (seed, tid , 0, &state[tid]);
}

__global__
void gpu_GA(curandState *d_state, double* population, double* fitness, const int populationSize, const int p, int numGenerations) {

    // Allocating shared memory for each island
    __shared__ double islandPop[SHMEN_SIZE];
    __shared__ double islandFitness[individualsPerIsland * 8];
    __shared__ double islandParent[SHMEN_SIZE];

    // Thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Every 5 generations migrate individuals to n+1 islands
    bool migrationFlag = false;
    if (numGenerations % 5 == 0) migrationFlag = true;
    
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

    // Tournament Selection
    selection(d_state, islandParent, islandPop, islandFitness, p, tid, individualsPerIsland);
    __syncthreads();

    // Arithmetic Crossover 
    arithmetic_crossover(d_state, islandParent, islandPop, p, tid, individualsPerIsland, crossoverProbability, alpha);
    __syncthreads();
    
    // TODO: Guassian Mutation (Look into implementing guassian mutation)
    mutation(d_state, islandPop, p, tid, lowerBound, upperBound, mutationProbability);
    __syncthreads();

    if (migrationFlag)
        migration(population, islandPop, islandFitness, p, tid, elitism, individualsPerIsland, islands);
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
    cudaMemset(d_fitness, 0, bytesFitness);

    // Copying population to device (intend to remove in the future)
    cudaMemcpy(d_population, h_population, bytesPopulation, cudaMemcpyHostToDevice);
    
    while (numGenerations > 0) {
        gpu_GA<<<islands, 
                 individualsPerIsland,
                 (SHMEN_SIZE*2) + (individualsPerIsland * 8)>>>
                 (d_state, d_population, d_fitness, populationSize, p, numGenerations);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError()); 
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
    float bounds[2] = {-5.12, 5.12};

    // GA parameters
    const int p = 10; // # of genes per individual
    const int populationSize = 2048; 
    int numGenerations = 100; 

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
    setup_kernel<<<islands, individualsPerIsland>>>(d_state, (unsigned long) t );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // GA
    parallelGA(population, fitness, populationSize, p, bytesPopulation, bytesFitness, numGenerations, d_state);

    printvec(fitness, populationSize);

    double *min = std::min_element(fitness, fitness + populationSize);

    // Find the minimum element
    cout << "\nMin Element = " << *min << std::endl;

    return 0;
}