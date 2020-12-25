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


void print2dvec(vector<vector<double>> vec) {

    for(auto& i : vec) {
        cout << "[ ";
        for(auto& j : i)
            cout << j << " ";
        cout << "]\n";
    }
}


int main() {
    

    // Optimization parameters for Sphere function
    vector<float> bounds{-5.12, 5.12};

    // GA parameters
    const int p = 30; // # of genes per individual
    const int populationSize = 100; 
    const int elitism = 1; 
    const int mating = ceil((populationSize-elitism)/2);
    const int tournamentSize = 2;
    int numGenerations = 5; 
    

    // Intialization for random number generator
    time_t t;
    srand((unsigned) time(&t));

    // Initialize Population 
    vector<vector<double>> population(populationSize, vector<double>(p, 0));
    initPopulation(population, bounds);
    print2dvec(population);

    // Evaluating Initial Population
    vector<double> fitness(populationSize, 0);
    evaluation(population, fitness, p);

    vector<vector<double>> parents(mating, vector<double>(p, 0));
    vector<double> offspring(p, 0);
    vector<vector<double>> temp_population(populationSize, vector<double>(p, 0));

    // Main GA loop
    while (numGenerations < 0) {


        tournamentSelection(parents, population, fitness, p, populationSize, mating, tournamentSize);
        print2dvec(parents);


        numGenerations--;
    }





    


    
    return 0;
}