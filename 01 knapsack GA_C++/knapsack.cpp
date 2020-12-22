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


void print2dvec(vector<vector<int>> vec) {

    for(auto& i : vec) {
        cout << "[ ";
        for(auto& j : i)
            cout << j << " ";
        cout << "]\n";
    }
}

void printvec(vector<int> vec) {

    for(auto& i : vec)
        cout << i << "\n";
}

int main() {



    // MHS Concepts values for knapsack problem
    vector<int> itemNumbers{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    vector<int> itemWeight{32, 40, 44, 20, 1, 29, 3, 13, 6, 39};
    vector<int> itemProfit{727, 736, 60, 606, 45, 370, 414, 880, 133, 820};
    const int maxWeight = 113;

    // Size of vector
    const int N = 10;
    size_t bytes = N * sizeof(int); 

    // Intialization for random number generator
    time_t t;
    srand((unsigned) time(&t));

    // Population parameters
    const int populationSize = 50;
    const int solutionsPerPopulation = N;

    // Initial Population 
    vector<vector<int>> population(solutionsPerPopulation, vector<int>(N, 0));
    initPopulation(population, solutionsPerPopulation, N);
    print2dvec(population);

    // Evaluating Initial Population for printing purposes
    vector<int> fitness(N, 0);
    evaluation(population, itemWeight, itemProfit, fitness, N, maxWeight);
    printvec(fitness);

    for (size_t i = 0; i < populationSize; i++) {
        
        // Evaluate Population on each iteration 
        memset(&fitness[0], 0, bytes);
        evaluation(population, itemWeight, itemProfit, fitness, N, maxWeight);

        //printvec(fitness);

        vector<vector<int>> parents(2, vector<int>(N, 0));
        selection(fitness, parents, population, N);

        //print2dvec(parents);

        vector<vector<int>> offsprings(2, vector<int>(N, 0));
        crossover(parents, offsprings, N);
        
        //print2dvec(offsprings);
        
        vector<vector<int>> mutants(2, vector<int>(N, 0));
        mutation(offsprings, mutants, N);

        //print2dvec(mutants);

        replacement(population, mutants, parents, fitness, itemWeight, itemProfit, N, maxWeight);

    }

    print2dvec(population);

    evaluation(population, itemWeight, itemProfit, fitness, N, maxWeight);

    printvec(fitness);


    return 0;    
}