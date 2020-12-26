#include <stdio.h> 
#include <stdlib.h>
#include <math.h> 
#include <cstring>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <cfloat>

#include "population.hpp"
#include "evaluate.hpp"
#include "selection.hpp"
#include "crossover.hpp"
#include "mutation.hpp"
#include "replacement.hpp"


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

void printvec(vector<double> vec) {

    for(auto& i : vec)
        cout << i << "\n";
}

int main() {
    

    // Optimization parameters for Sphere function
    vector<float> bounds{-5.12, 5.12};

    // GA parameters
    const int p = 30; // # of genes per individual
    const int populationSize = 100; 
    const int elitism = 4; 
    const int mating = ceil((populationSize)/2);
    const int tournamentSize = 5;
    int numGenerations = 300000; 
    const float crossoverProbability = 0.9f;
    const float mutationProbability = 0.01f;
    
    // Intialization for random number generator
    time_t t;
    srand((unsigned) time(&t));

    // Initialize Population 
    vector<vector<double>> population(populationSize, vector<double>(p, 0));
    initPopulation(population, bounds);

    // Evaluating Initial Population
    vector<double> fitness(populationSize, 0);
    evaluation(population, fitness, p);
    printvec(fitness);

    vector<vector<double>> parents(populationSize, vector<double>(p, 0));
    vector<vector<double>> temp_population(populationSize, vector<double>(p, 0));

    

    // Main GA loop
    while (numGenerations > 0) {

        tournamentSelection(parents, population, fitness, p, populationSize, tournamentSize);

        crossover(temp_population, parents, p, crossoverProbability, mating);

        mutation(temp_population, bounds, p, populationSize, mutationProbability);

        replacement(population, temp_population, fitness, p, populationSize, elitism);

        evaluation(population, fitness, p);

        numGenerations--;
    }

    // print2dvec(population);
    cout << "new population" << std::endl;
    printvec(fitness);


    
    return 0;
}