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
#define islands 4
#define individualsPerIsland 32/islands

#define SHMEN_SIZE (10 * individualsPerIsland * 8)

#define tournamentSize 2
#define crossoverProbability 0.9f
#define mutationProbability 0.05f
#define alpha 0.5f

__global__
void setup_kernel (curandState* state, unsigned long seed )
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init (seed, id , seed, &state[id]);
}

__device__
double square(double x) { return x * x; }

__global__
void gpu_GA(curandState *d_state, double* population, double* fitness, const int populationSize, const int p) {

    // Allocating shared memory for each island
    __shared__ double islandPop[SHMEN_SIZE];
    __shared__ double islandFitness[individualsPerIsland * 8];
    __shared__ double islandParent[SHMEN_SIZE];

    // Min & Max Index
    int minIndex, maxIndex;

    // Thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    islandFitness[threadIdx.x] = 0;
    __syncthreads();

    // Load elements into shared memory
    for (unsigned int i = 0; i < p; i++)
        islandPop[threadIdx.x * p + i] = population[tid * p + i];
    __syncthreads();

    // Evaluation for each individual
    for (unsigned int i = 0; i < p; i++)
        islandFitness[threadIdx.x] += square(islandPop[threadIdx.x * p + i]);
    __syncthreads();

    //printf("%f \n",  islandFitness[threadIdx.x]);

    // Individuals selected for tournament
    unsigned int tournamentPool[tournamentSize];
    for (unsigned int i = 0; i < tournamentSize; i++) {
        
        float myrand = curand_uniform(&d_state[tid]);
        myrand *= (individualsPerIsland-1+0.999999);
        int value = (int)truncf(myrand);

        tournamentPool[i] = value;

    }

    // select best individual in tournament (
    unsigned int parentIndex = islandFitness[tournamentPool[0]] > islandFitness[tournamentPool[1]] ? tournamentPool[0] : tournamentPool[1];
    
    // Copying best individual from tournament to island parent 
    for (unsigned int i = 0; i < p; i++)
        islandParent[threadIdx.x * p + i] = islandPop[parentIndex * p + i];
    __syncthreads();

    // printf("tid:%d %d \n", tid, parentIndex);

    
    // Arithmetic Crossover 
    float myrand = curand_uniform(&d_state[tid]);

    if (myrand < crossoverProbability) {
        if (threadIdx.x < (islands/2)) {

            for (unsigned int i = 0; i < p; i++) {
                islandPop[threadIdx.x * p + i] = (alpha * islandParent[threadIdx.x * p + i]) + (1-alpha) * islandParent[(threadIdx.x+(islands/2)) * p + i];
                islandPop[(threadIdx.x+(islands/2)) * p + i] = (alpha * islandParent[threadIdx.x * p + i]) + (1-alpha) * islandParent[(threadIdx.x+(islands/2)) * p + i];
            }
        } 
    }
    __syncthreads();

    // TODO: Guassian Mutation (Look into implementing guassian mutation)

    for (unsigned int i = 0; i < p; i++) {
        float myrand = curand_uniform(&d_state[tid]);
        if (myrand < mutationProbability) 
        islandPop[threadIdx.x * p + i] = -5.12 + (curand_uniform(&d_state[tid])) * (5.12 - -5.12);
    }
    __syncthreads();
    // TODO: Replacement ?

    // TODO: Migration
    // Go over strategies with Jeremy.
    // Propose current strategy. (Can't think of non-strided memory access)
    // use thrust to identify max fitness individuals
    // use thrust to identify min fitness individuals
    double *minVal = thrust::min_element(thrust::device, islandFitness, islandFitness + individualsPerIsland);
    double *maxVal = thrust::max_element(thrust::device, islandFitness, islandFitness + individualsPerIsland);
    __syncthreads();

    if (threadIdx.x == 0) {
        printf("min: %f \n", *minVal);
        printf("max: %f \n", *maxVal);
    }

    if (islandFitness[threadIdx.x] == *minVal)
        minIndex = threadIdx.x;
    if (islandFitness[threadIdx.x] == *maxVal)
        maxIndex = threadIdx.x;

    if (threadIdx.x == 0) {
        printf("min: %f \n", *minVal);
        printf("max: %f \n", *maxVal);
    }
        

    // Copy back to global memory
    for (unsigned int i = 0; i < p; i++)
        population[tid * p + i] = islandPop[threadIdx.x * p + i];

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
                 (SHMEN_SIZE*2) + (individualsPerIsland * 8)>>>(d_state, d_population, d_fitness, populationSize, p);
        cudaDeviceSynchronize(); 
        numGenerations--;

    }

    // Copy population and fitness back to host
    cudaMemcpy(h_population, d_population, bytesPopulation, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fitness, d_fitness, bytesFitness, cudaMemcpyDeviceToHost);

    cudaFree(d_population);
    cudaFree(d_fitness);

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
    const int populationSize = 32; 
    //const int elitism = 5; 
    const int mating = ceil((populationSize)/2);
    int numGenerations = 1; 

    // Intialization for random number generator
    time_t t;
    time(&t);

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

    //print2dvec(population, populationSize, p);

    // cuRand setup
    curandState *d_state;
    cudaMalloc(&d_state, islands*individualsPerIsland*sizeof( curandState ) );
    setup_kernel<<<islands,individualsPerIsland>>>(d_state, (unsigned long) t );

    // GA
    parallelGA(population, fitness, populationSize, p, bytesPopulation, bytesFitness, numGenerations, d_state);

    //printvec(fitness, populationSize);

    //print2dvec(population, populationSize, p);
    
    return 0;
}