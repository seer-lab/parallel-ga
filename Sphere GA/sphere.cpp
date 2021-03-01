#include <stdio.h> 
#include <stdlib.h>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <iostream>
#include "replacement.h"
#include "mutation.h"
#include "crossover.h"
#include "selection.h"
#include "evaluation.h"
#include "population.h"

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

void copy(vector<vector<double>> &vec, vector<vector<double>> mutants, vector<vector<double>> parents, int index, const int p) {

    for (vector<int>::size_type j = 0; j != p; j++) {

        vec[index][j] = parents[0][j];
        vec[index+24][j] = parents[1][j];
        vec[index+(24*2)][j] = mutants[0][j];
        vec[index+(24*3)][j] = mutants[1][j];

    }

}

int main() {


    // Optimization Parameters for Sphere function
    vector<float> bounds{-5.12, 5.12};

    // Number of genes per Individual
    const int p = 30;
    size_t bytes = p * sizeof(double); 

    // Intialization for random number generator
    time_t t;
    srand((unsigned) time(&t));

    // Population parameters
    const int populationSize = 100;

    // Initialize Population 
    vector<vector<double>> population(populationSize, vector<double>(p, 0));
    initPopulation(population, bounds);
    //print2dvec(population);

    // Evaluating Initial Population
    vector<double> fitness(populationSize, 0);
    evaluation(population, fitness, p);
    //printvec(fitness);

    vector<vector<double>> parents(2, vector<double>(p, 0));
    vector<vector<double>> offsprings(2, vector<double>(p, 0));
    vector<vector<double>> mutants(2, vector<double>(p, 0));
    vector<vector<double>> temp_population(populationSize, vector<double>(p, 0));

    

    for (size_t i = 0; i < 1000; i++) {

        for (size_t j = 0; j < (populationSize/4-1); j++) {
            
            //memset(&parents[0], 0, parents.size() * sizeof(parents[0]));
            selection(parents, fitness, population, populationSize, p);

            //print2dvec(parents);

            //memset(&offsprings[0], 0, 2*p*sizeof(double));
            crossover(parents, offsprings, p);
            
            //print2dvec(offsprings);

            //memset(&mutants[0], 0, 2*p*sizeof(double));
            mutation(offsprings, bounds, mutants, p);

            //print2dvec(mutants);

            copy(temp_population, mutants, parents, j, p);
        }

        replacement(temp_population, population, fitness, populationSize, p);
        
        // Evaluate Population on each iteration 
        //memset(&fitness[0], 0, fitness.size() * sizeof(fitness[0]));
        evaluation(population, fitness, p);

        //memset(&temp_population[0], 0, temp_population.size() * sizeof(temp_population[0]));

    }

    print2dvec(population);
    evaluation(population, fitness, p);
    printvec(fitness);


    return 0;    
}