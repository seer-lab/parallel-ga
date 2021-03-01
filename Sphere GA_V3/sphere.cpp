#include <cmath> 
#include <vector>
#include <iostream>

#include "replacement.h"
#include "mutation.h"
#include "crossover.h"
#include "selection.h"
#include "evaluation.h"
#include "population.h"

using std::vector;
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
    vector<float> bounds{-5.12, 5.12};

    // GA parameters
    const int p = 10; // # of genes per individual
    const int populationSize = 16384; 
    const int elitism = 0; 
    const int mating = ceil((populationSize)/2);
    const int tournamentSize = 2;
    int numGenerations = 1000; 
    const float crossoverProbability = 0.9f;
    const float mutationProbability = 0.05f;


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


    // For testing purposes

    // print2dvec(population, populationSize, p);
    // printvec(fitness, populationSize);

    
    // Main GA loop
    while (numGenerations > 0) {

        tournamentSelection(parents, population, fitness, p, populationSize, tournamentSize);
        crossover(temp_population, parents, p, crossoverProbability, mating);
        mutation(temp_population, bounds, p, populationSize, mutationProbability);
        replacement(population, temp_population, fitness, p, populationSize, elitism);
        evaluation(population, fitness, populationSize, p);

        numGenerations--;
    }
    
    // For testing purposes

    // cout << "new population" << std::endl;
    printvec(fitness, populationSize);

    return 0;
}