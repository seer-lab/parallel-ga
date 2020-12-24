#include <stdio.h> 
#include <stdlib.h>
#include <math.h> 
#include <cstring>
#include <vector>
#include <cstdlib>
#include <iostream>

#include "population.hpp"
#include "evaluate.hpp"
#include "selection.hpp"

using std::vector;
using std::cout;



int main() {
    

    // Optimization parameters for Sphere function
    vector<float> bounds{-5.12, 5.12};

    // GA parameters
    const int p = 30; // # of genes per individual
    const int populationSize = 100; 
    const int survive = 1; // Elitism  
    const int keep = ceil((populationSize-survive)/2); // Number of matings
    int numGenerations = 300000; 

    // Intialization for random number generator
    time_t t;
    srand((unsigned) time(&t));

    // Initialize Population 
    vector<vector<double>> population(populationSize, vector<double>(p, 0));
    initPopulation(population, bounds);

    // Evaluating Initial Population
    vector<double> fitness(populationSize, 0);
    evaluation(population, fitness, p);

    vector<vector<double>> parents(keep, vector<double>(p, 0));
    vector<double> offspring(p, 0);
    vector<vector<double>> temp_population(populationSize, vector<double>(p, 0));

    // TODO implement the main GA loop refer to paper "An Introduction to Genetic Algorithms" and ECE457A
    while (numGenerations < 0) {
        








        numGenerations--;
    }





    


    
    return 0;
}