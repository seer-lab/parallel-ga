#include <stdio.h> 
#include <stdlib.h>
#include <cstring>
#include <cstdlib>
#include <math.h>
#include <algorithm>

#include "include/utils.h" // For cuda imports

#include "include/migration.cuh"
#include "include/mutation.cuh"
#include "include/crossover.h"
#include "include/selection.cuh"
#include "include/evaluation.cuh"
#include "include/population.h"

using std::cout;
using std::endl;

// Island Parameters
#define islands 64
#define individualsPerIsland 8192/islands

#define elitism 2
#define tournamentSize 10
#define crossoverProbability 0.9f
#define mutationProbability 0.001f
#define alpha 0.25f
#define numGen 1000

#define lowerBound -5.12
#define upperBound 5.12


__global__
void setup_kernel (curandState* state, unsigned long seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init (seed, tid , 0, &state[tid]);
}

__global__
void gpu_GA_pre_crossover(curandState *d_state, double* population, double* fitness, double* parents, const int populationSize, const int p, int numGenerations) { 

    // Thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (numGenerations == numGen) {
        evaluation(fitness, population, p, tid);
        __syncthreads();
    }
    
    // Tournament Selection
    selection(d_state, parents, population, fitness, p, tid, individualsPerIsland);
    __syncthreads();
}   

__global__
void gpu_GA_post_crossover(curandState *d_state, double* population, double* fitness, double* parents, const int populationSize, const int p, int numGenerations) {

        // Thread ID
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        // Every 5 generations migrate individuals to n+1 islands
        bool migrationFlag = false;
        if (numGenerations % 5 == 0) migrationFlag = true;

        // TODO: Guassian Mutation (Look into implementing guassian mutation)
        mutation(d_state, population, p, tid, lowerBound, upperBound, mutationProbability);
        __syncthreads();

        // TODO: Migration
        if (migrationFlag)
            migration(population, fitness, p, tid, elitism, individualsPerIsland, islands);
        __syncthreads();

        fitness[tid] = 0;
        __syncthreads();

        // Evaluation for each individual
        evaluation(fitness, population, p, tid);
        __syncthreads();

}

void parallelGA(double* h_population, 
                double* h_fitness, 
                const int populationSize, 
                const int p, 
                size_t bytesPopulation, 
                size_t bytesFitness,
                int numGenerations,
                curandState *&d_state,
                const int mating,
                time_t t) {
    
    // Allocating device memory
    double *d_population, *d_fitness, *d_parents, *h_parents;

    h_parents = (double*)malloc(bytesPopulation);

    cudaMalloc(&d_population, bytesPopulation);
    cudaMalloc(&d_fitness, bytesFitness);
    cudaMalloc(&d_parents, bytesPopulation);

    cudaMemset(d_fitness, 0, bytesFitness);
    cudaMemset(d_parents, 0, bytesPopulation);
    
    // Copying population to device (intend to remove in the future)
    cudaMemcpy(d_population, h_population, bytesPopulation, cudaMemcpyHostToDevice);
    
    while (numGenerations > 0) {

        setup_kernel<<<islands,individualsPerIsland>>>(d_state, (unsigned long) t );
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());  

        gpu_GA_pre_crossover<<<islands, 
                 individualsPerIsland>>>
                 (d_state, d_population, d_fitness, d_parents, populationSize, p, numGenerations);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());  
        
        // Copy population and fitness back to host
        cudaMemcpy(h_population, d_population, bytesPopulation, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_parents, d_parents, bytesPopulation, cudaMemcpyDeviceToHost);

        // CPU Crossover
        arithmetic_crossover(h_population, h_parents, p, crossoverProbability, mating, alpha);

        // Copying population to device (intend to remove in the future)
        cudaMemcpy(d_population, h_population, bytesPopulation, cudaMemcpyHostToDevice);
        cudaMemcpy(d_parents, h_parents, bytesPopulation, cudaMemcpyHostToDevice);

        gpu_GA_post_crossover<<<islands, 
                 individualsPerIsland>>>
                 (d_state, d_population, d_fitness, d_parents, populationSize, p, numGenerations);
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
    const int p = 1024; // # of genes per individual
    const int populationSize = 8192; 
    const int mating = ceil((populationSize)/2);
    int numGenerations = 1000; 

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

    // GA
    parallelGA(population, fitness, populationSize, p, bytesPopulation, bytesFitness, numGenerations, d_state, mating, t);

    printvec(fitness, populationSize);

    double *min = std::min_element(fitness, fitness + populationSize);

    // Find the minimum element
    cout << "\nMin Element = " << *min << endl;

    return 0;
}