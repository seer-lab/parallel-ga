#include <stdlib.h> // for rand
#include <vector> 
#include <cfloat> // For double_max
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>

#include "include/replacement.h"
#include "include/mutation.h"
#include "include/crossover.h"
#include "include/selection.h"
#include "include/population.h"
#include "include/evaluation.h"

using std::cout;

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
    float bounds[2] = {-600.0f, 600.0f};

    // GA parameters
    const int p = 40; // # of genes per individual
    const int populationSize = 1024; 
    const int elitism = 2; 
    const int mating = ceil((populationSize)/2);
    const int tournamentSize = 6;
    int numGenerations = 100; 
    const float crossoverProbability = 0.9f;
    const float mutationProbability = 0.01f;
    const float alpha = 0.25f;
    const int nc = 2;

    // Intialization for random number generator
    time_t t;
    srand((unsigned) time(&t));

    // Vector size
    size_t bytesPopulation = p * populationSize * sizeof(double);
    size_t bytesFitness = populationSize * sizeof(double);

    // Initilize vectors
    double *population, *fitness, *parents, *temp_population;

    // Allocate memory
    population = (double*)malloc(bytesPopulation);
    fitness = (double*)malloc(bytesFitness);
    parents = (double*)malloc(bytesPopulation);
    temp_population = (double*)malloc(bytesPopulation);

    // Initialize Population 
    initPopulation(population, bounds, populationSize, p);
    evaluation(population, fitness, populationSize, p);
    
    // Main GA loop
    while (numGenerations > 0) {

        tournamentSelection(parents, population, fitness, p, populationSize, tournamentSize);
        simulated_binary_crossover(temp_population, parents, p, crossoverProbability, mating, nc);
        mutation(temp_population, bounds, p, populationSize, mutationProbability);
        replacement(population, temp_population, fitness, p, populationSize, elitism);
        evaluation(population, fitness, populationSize, p);

        numGenerations--;
    }
    
    // For testing purposes
    printvec(fitness, populationSize);

    double *min = std::min_element(fitness, fitness + populationSize);

    // Find the minimum element
    cout << "\nMin Element = " << *min << std::endl;
    return 0;
}