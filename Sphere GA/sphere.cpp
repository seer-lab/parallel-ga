#include <stdio.h> 
#include <stdlib.h>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <iostream>

#include "population.h"
#include "evaluate.h"

using std::vector;
using std::cout;



int main() {
    


    // Optimization Parameters for Sphere function
    vector<float> bounds{-5.12, 5.12};

    // # of genes per individual
    const int p = 30;
    size_t bytes = p * sizeof(double);

    // Size of population on each generation
    const int populationSize = 100;

    // Intialization for random number generator
    time_t t;
    srand((unsigned) time(&t));

    // Initialize Population 
    vector<vector<double>> population(populationSize, vector<double>(p, 0));
    initPopulation(population, bounds);

    // Evaluating Initial Population
    vector<double> fitness(populationSize, 0);
    evaluation(population, fitness, p);

    vector<vector<double>> parents(2, vector<double>(p, 0));
    vector<vector<double>> offsprings(2, vector<double>(p, 0));
    vector<vector<double>> mutants(2, vector<double>(p, 0));
    vector<vector<double>> temp_population(populationSize, vector<double>(p, 0));

    
    return 0;
}