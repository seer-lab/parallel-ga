#include <stdio.h> 
#include <stdlib.h>
#include <cmath> 
#include <cstring>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <cfloat>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include "utils.h"
#include "timer.h"

using std::vector;
using std::cout;


#define warp 128

void initPopulation(double* population, std::vector<float> bounds, const int row, const int col) {

    for (unsigned int i = 0; i < row; i++)
        for (unsigned int j = 0; j < col; j++)
            *(population + i*col + j) = bounds[0] + ((double)rand() / RAND_MAX) * (bounds[1] - bounds[0]);

}

__device__
double square(double x) { return x * x; }

__global__
void gpu_eval(double *p, double *f, int numRows, int numCols) {

    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid >= numRows)
        return;

    for (unsigned int i = 0; i < numCols; i++)
        f[tid] += square(p[tid * numCols + i]);

}

void evaluation(double* h_population, double* h_fitness, const int row, const int col, size_t bytesPopulation, size_t bytesFitness) {

    
    double *d_population, *d_fitness;

    cudaMalloc(&d_population, bytesPopulation);
    cudaMalloc(&d_fitness, bytesFitness);

    cudaMemcpy(d_population, h_population, bytesPopulation, cudaMemcpyHostToDevice);

    int TB_SIZE = row/warp;

    gpu_eval<<<warp, TB_SIZE>>>(d_population, d_fitness, row, col);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    cudaMemcpy(h_fitness, d_fitness, bytesFitness, cudaMemcpyDeviceToHost);

    cudaFree(d_population);
    cudaFree(d_fitness);

}

double square2(double x) { return x * x; }

double sphere(double* individual, int index, const int p) { 

    double sum = 0.0;
    for (std::vector<int>::size_type i = 0; i != p; i++)
        sum += square2(individual[index * p + i]);

    return sum; 
}

void evaluation(double* population, double* fitness, const int populationSize, const int p) {


    for (unsigned int i = 0; i != populationSize; i++) 
        fitness[i] = sphere(population, i, p);

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

int bestIndividual(double *a, std::vector<int> b, int n) {
    
    unsigned int i, index = b[0];
    double diff = DBL_MAX;

    for (i = 0; i < n; i++) {
        double absVal = abs(a[b[i]]);
        if (absVal < diff) {
            index = b[i];
            diff = absVal;
        } else if (absVal == diff && a[b[i]] > 0 && a[index] < 0) {
            index = b[i];
        }
    }

    return index;
}

int bestIndividual(double *a, int n) {

    unsigned int i, index = 0;
    double diff = DBL_MAX;
    
    for (i = 0; i < n; i++) {
        double absVal = abs(a[i]);
        if (absVal < diff) {
            index = i;
            diff = absVal;
        } else if (absVal == diff && a[i] > 0 && a[index] < 0) {
            index = i;
        }
    }

    return index;
}


void tournamentSelection(double *parents, double* population, double *fitness, const int p, const int populationSize, const int tournamentSize) {

    std::vector<int> tournamentPool(tournamentSize, 0);

    unsigned int count = 0;
    int parentIndex = 0;
    while (count < populationSize) {
        
        // selecting individuals for tournament
        for (std::vector<int>::size_type i = 0; i != tournamentSize; i++) 
            tournamentPool[i] = rand() % populationSize; 

        parentIndex = bestIndividual(fitness, tournamentPool, tournamentSize);

        
        for (std::vector<int>::size_type i = 0; i != p; i++) 
            parents[count * p + i] = population[parentIndex * p + i];

        count++;
    }

}


void crossover(double *temp_population, double *parents, const int p, const float crossoverProbability, const int mating) {

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

void mutation(double *temp_population, std::vector<float> bounds, const int p, const int populationSize, const float mutationProbability) {


    for(unsigned int i = 0; i < populationSize; i++) {
        for(unsigned int j = 0; j < p; j++) {
            if (((float)rand())/RAND_MAX < mutationProbability) 
                temp_population[i * p + j] = bounds[0] + ((double)rand() / RAND_MAX) * (bounds[1] - bounds[0]);
        }
    }

}

void replacement (double *population, double* temp_population, double *fitness, const int p, const int populationSize, const int elitism) {

    std::vector<int> minIndex(elitism, 0);


    for(unsigned int i = 0; i < elitism; i++) {
        minIndex[i] = bestIndividual(fitness, populationSize);
        fitness[minIndex[i]] = DBL_MAX;
    }
    
    for(unsigned int i = 0; i < elitism; i++)
        for(unsigned int j = 0; j < p; j++)
            population[i * p + j] = population[minIndex[i] * p + j];

    for(unsigned int i = elitism; i < populationSize; i++)
        for(unsigned int j = 0; j < p; j++)
            population[i * p + j] = temp_population[i * p + j];

}

int main() {

    // Optimization parameters for Sphere function
    vector<float> bounds{-5.12, 5.12};

    // GA parameters
    const int p = 128; // # of genes per individual
    const int populationSize = 32768; 
    const int elitism = 5; 
    const int mating = ceil((populationSize)/2);
    const int tournamentSize = 5;
    int numGenerations = 100; 
    const float crossoverProbability = 0.9f;
    const float mutationProbability = 0.01f;


    // Intialization for random number generator
    time_t t;
    srand((unsigned) time(&t));

    // Vector size
    size_t bytesPopulation = p * populationSize * sizeof(double);
    size_t bytesFitness = populationSize * sizeof(double);

    // Initilize vectors
    double *population, *fitness, *parents, *temp_population, *population_h, *fitness_h, *parents_h, *temp_population_h;

    // Allocate memory
    population = (double*)malloc(bytesPopulation);
    population_h = (double*)malloc(bytesPopulation);
    fitness = (double*)malloc(bytesFitness);
    fitness_h = (double*)malloc(bytesFitness);
    parents = (double*)malloc(bytesPopulation);
    parents_h = (double*)malloc(bytesPopulation);
    temp_population = (double*)malloc(bytesPopulation);
    temp_population_h = (double*)malloc(bytesPopulation);

    //cout << "GPU Part\n";
    // Initialize Population 
    initPopulation(population, bounds, populationSize, p);

    evaluation(population, fitness, populationSize, p, bytesPopulation, bytesFitness);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    //cout << "GPU Population\n";

    //print2dvec(population, populationSize, p);

    //cout << "GPU Fitness\n";
    // printvec(fitness, populationSize);
    

    // Main GA loop
    while (numGenerations > 0) {

        tournamentSelection(parents, population, fitness, p, populationSize, tournamentSize);
        crossover(temp_population, parents, p, crossoverProbability, mating);
        mutation(temp_population, bounds, p, populationSize, mutationProbability);
        replacement(population, temp_population, fitness, p, populationSize, elitism);
        evaluation(population, fitness, populationSize, p, bytesPopulation, bytesFitness);

        numGenerations--;
    }
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    //cout << "new fitness GPU" << std::endl;
    //printvec(fitness, populationSize);

    /*
    //cout << "CPU PART \n";
    initPopulation(population_h, bounds, populationSize, p);

    //cout << "CPU Population\n";

    //print2dvec(population_h, populationSize, p);

    numGenerations = 100; 

    evaluation(population_h, fitness_h, populationSize, p);

    //cout << "CPU Fitness\n";
    //printvec(fitness_h, populationSize);

    while (numGenerations > 0) {

        tournamentSelection(parents_h, population_h, fitness_h, p, populationSize, tournamentSize);
        crossover(temp_population_h, parents_h, p, crossoverProbability, mating);
        mutation(temp_population_h, bounds, p, populationSize, mutationProbability);
        replacement(population_h, temp_population_h, fitness_h, p, populationSize, elitism);
        evaluation(population_h, fitness_h, populationSize, p);

        numGenerations--;
    }

    //cout << "new fitness CPU" << std::endl;
    //printvec(fitness_h, populationSize);
    */
    return 0;
}
