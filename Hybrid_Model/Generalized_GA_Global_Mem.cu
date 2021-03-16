#include <stdio.h> 
#include <stdlib.h>
#include <cmath> 
#include <cstring>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <cfloat>
#include <math.h>
#include <algorithm>

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
#define individualsPerIsland 8192/islands

#define elitism 2
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
void evaluation(double *fitness, double *population, const int p, int tid) {

    for (unsigned int i = 0; i < p; i++)
        fitness[tid] += square(population[tid * p + i]);
}


__device__
int bestIndividual(double *a, int *b, int n) {
    
    int i, index = b[0];
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
void selection(curandState *d_state, double *parents, double *population, double *fitness, const int p, int tid) {
    // Individuals selected for tournament
    int tournamentPool[tournamentSize];

    for (unsigned int i = 0; i < tournamentSize; i++) {
        float myrand = curand_uniform(&d_state[tid]);
        myrand *= ((individualsPerIsland*blockIdx.x+individualsPerIsland) - (individualsPerIsland*blockIdx.x)-1+0.999999);
        myrand += (individualsPerIsland*blockIdx.x);
        int value = (int)truncf(myrand);

        tournamentPool[i] = value;
    }

    // select best individual in tournament 
    int parentIndex = bestIndividual(fitness, tournamentPool, tournamentSize);

    // Copying best individual from tournament to island parent 
    for (unsigned int i = 0; i < p; i++)
        parents[tid * p + i] = population[parentIndex * p + i];

}

/*
__device__
void crossover(curandState *d_state, double *parents, double *population, const int p, int tid) {
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
*/

__device__
void mutation(curandState *d_state, double *population, const int p, int tid) {
    for (unsigned int i = 0; i < p; i++) {
        float myrand = curand_uniform(&d_state[tid]);
        if (myrand < mutationProbability) 
            population[tid * p + i] = -5.12 + (curand_uniform(&d_state[tid])) * (5.12 - -5.12);
    }
}

// Helper function for elitism
// Determines best individual in the population
__device__
int bestIndividual(double *a, int s, int n) {
    unsigned int i, index = 0;
    double diff = DBL_MAX;
    
    for (i = s; i < n; i++) {
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
void migration(double* population, double *fitness, const int p, int tid) {

    unsigned int minIndex;

    if (tid % islands < elitism) {
        minIndex = bestIndividual(fitness, tid-threadIdx.x, (tid-threadIdx.x)+individualsPerIsland);
        fitness[minIndex] = DBL_MAX;
    }

    if (tid % islands < elitism) {
        for (unsigned int i = 0; i < p; i++) {
            population[(((blockIdx.x+1)%islands) * blockDim.x + threadIdx.x) * p + i] = population[minIndex * p + i];
        }
    }
}

__global__
void gpu_GA_pre_crossover(curandState *d_state, double* population, double* fitness, double* parents, const int populationSize, const int p, int numGenerations) { 

    // Thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (numGenerations == numGen)
        evaluation(fitness, population, p, tid);
    __syncthreads();
    
    // Tournament Selection
    selection(d_state, parents, population, fitness, p, tid);
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
        mutation(d_state, population, p, tid);
        __syncthreads();
    
        // TODO: Migration
        if (migrationFlag)
            migration(population, fitness, p, tid);
        __syncthreads();
    
        fitness[tid] = 0;
        __syncthreads();
    
        // Evaluation for each individual
        evaluation(fitness, population, p, tid);
        __syncthreads();

}

void initPopulation(double* population, std::vector<float> bounds, const int row, const int col) {

    for (unsigned int i = 0; i < row; i++)
        for (unsigned int j = 0; j < col; j++)
            *(population + i*col + j) = bounds[0] + ((double)rand() / RAND_MAX) * (bounds[1] - bounds[0]);

}
// Single-point crossover
void crossover(double *temp_population, double *parents, const int p, const int mating) {
    int crossoverPoint = 0;
    for (unsigned int i = 0; i < mating; i++) {
        if (((float)rand())/RAND_MAX < crossoverProbability) {
            crossoverPoint = rand() % p;

            for (unsigned int j = 0; j < crossoverPoint; j++) {
                temp_population[i * p + j] = parents[i * p + j];
                temp_population[(i+mating) * p + j] = parents[(i+mating) * p + j];
            }

            for (unsigned int j = crossoverPoint; j < p; j++) {
                temp_population[i * p + j] = parents[(i+mating) * p + j];
                temp_population[(i+mating) * p + j] = parents[i * p + j];
            }
        } else {
            for (unsigned int j = 0; j < p; j++) {
                temp_population[i * p + j] = parents[i * p + j];
                temp_population[(i+mating) * p + j] = parents[(i+mating) * p + j];
            }
        }
    }
}

void parallelGA(double* h_population, 
                double* h_fitness, 
                const int populationSize, 
                const int p, 
                size_t bytesPopulation, 
                size_t bytesFitness,
                int numGenerations,
                curandState *&d_state,
                const int mating) {
    
    // Allocating device memory
    double *d_population, *d_fitness, *d_parents, *h_parents;

    h_parents = (double*)malloc(bytesPopulation);

    cudaMalloc(&d_population, bytesPopulation);
    cudaMalloc(&d_fitness, bytesFitness);
    cudaMalloc(&d_parents, bytesPopulation);
    
    // Copying population to device (intend to remove in the future)
    cudaMemcpy(d_population, h_population, bytesPopulation, cudaMemcpyHostToDevice);
    
    while (numGenerations > 0) {
        gpu_GA_pre_crossover<<<islands, 
                 individualsPerIsland>>>
                 (d_state, d_population, d_fitness, d_parents, populationSize, p, numGenerations);
        cudaDeviceSynchronize(); 
        
        // Copy population and fitness back to host
        cudaMemcpy(h_population, d_population, bytesPopulation, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_parents, d_parents, bytesPopulation, cudaMemcpyDeviceToHost);

        // CPU Crossover
        crossover(h_population, h_parents, p, mating);

        // Copying population to device (intend to remove in the future)
        cudaMemcpy(d_population, h_population, bytesPopulation, cudaMemcpyHostToDevice);
        cudaMemcpy(d_parents, h_parents, bytesPopulation, cudaMemcpyHostToDevice);

        gpu_GA_post_crossover<<<islands, 
                 individualsPerIsland>>>
                 (d_state, d_population, d_fitness, d_parents, populationSize, p, numGenerations);
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
    const int p = 30; // # of genes per individual
    const int populationSize = 8192; 
    const int mating = ceil((populationSize)/2);
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
    parallelGA(population, fitness, populationSize, p, bytesPopulation, bytesFitness, numGenerations, d_state, mating);

    printvec(fitness, populationSize);

    double *min = std::min_element(fitness, fitness + populationSize);

    // Find the minimum element
    cout << "\nMin Element = "
         << *min << std::endl;

    return 0;
}