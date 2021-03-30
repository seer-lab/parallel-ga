#include <stdlib.h> // for rand
#include <vector> 
#include <cfloat> // For double_max
#include <algorithm>

#include "include/utils.h" // For cuda 

#include "include/replacement.h"
#include "include/mutation.h"
#include "include/crossover.h"
#include "include/selection.h"
#include "include/population.h"
#include "include/evaluation.cuh"

using std::vector;
using std::cout;

#define warp 64
#define alpha 0.25

// Helper function for printing the population and fitness
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
    float bounds[2] = {-32.768, 32.768};

    // GA parameters
    const int p = 128; // # of genes per individual
    const int populationSize = 8192; 
    const int elitism = 6; 
    const int mating = ceil((populationSize)/2);
    const int tournamentSize = 2;
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
    double *population, *fitness, *parents, *temp_population, *d_population, *d_fitness;

    // Allocate memory
    population = (double*)malloc(bytesPopulation);
    fitness = (double*)malloc(bytesFitness);
    parents = (double*)malloc(bytesPopulation);
    temp_population = (double*)malloc(bytesPopulation);

    // Memory allocation for device variables
    cudaMalloc(&d_population, bytesPopulation);
    cudaMalloc(&d_fitness, bytesFitness);

    cudaMemset(d_fitness, 0, bytesFitness);

    // Initialize Population 
    initPopulation(population, bounds, populationSize, p);

    evaluation(warp, population, fitness, d_population, d_fitness, populationSize, p, bytesPopulation, bytesFitness);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // Main GA loop
    while (numGenerations > 0) {

        tournamentSelection(parents, population, fitness, p, populationSize, tournamentSize);
        arithmetic_crossover(temp_population, parents, p, crossoverProbability, mating, alpha);
        mutation(temp_population, bounds, p, populationSize, mutationProbability);
        replacement(population, temp_population, fitness, p, populationSize, elitism);
        evaluation(warp, population, fitness, d_population, d_fitness, populationSize, p, bytesPopulation, bytesFitness);

        numGenerations--;
    }

    cudaFree(d_population);
    cudaFree(d_fitness);

    printvec(fitness, populationSize);

    double *min = std::min_element(fitness, fitness + populationSize);

    // Find the minimum element
    cout << "\nMin Element = " << *min << std::endl;

    return 0;
}