#include <stdio.h> 
#include <stdlib.h>
#include "selection.h"
#include "evaluation.h"
#include "crossover.h"
#include "mutation.h"
#include "replacement.h"
#include "population.h"



int main() {

    // MHS Concepts values for knapsack problem
    const int itemNumbers[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const int itemWeight[10] = {32, 40, 44, 20, 1, 29, 3, 13, 6, 39};
    const int itemProfit[10] = {727, 736, 60, 606, 45, 370, 414, 880, 133, 820};
    const int maxWeight = 113;
    const int length = sizeof(itemNumbers)/sizeof(itemNumbers[0]);

    // Intialization for random number generator
    time_t t;
    srand((unsigned) time(&t));

    /*
    printf("Item Number: \tItem Weight: \tItem Profit: \n");

    for (int i = 0; i < length; i++) {
        printf("%d\t\t%d\t\t%d\n", itemNumbers[i], itemWeight[i], itemProfit[i]);
    }
    */

    const int populationSize = 50;
    const int solutionsPerPopulation = 10;
    int population[solutionsPerPopulation][length];
    memset(population, 0, solutionsPerPopulation*length*sizeof(int));

    initPopulation(population, solutionsPerPopulation, length);

    printf("Initial Population:\n");

    for(int i = 0; i < solutionsPerPopulation; i++) {
        printf("[ ");
        for (int j = 0; j < length; j++) {  
            printf("%d ", population[i][j]);    
        }  
        printf("]\n");
    }  

    int fitness[length];
    memset(fitness, 0, length*sizeof(int));

    evaluation(population, itemWeight, itemProfit, fitness, length, maxWeight);
    
    printf("Initial Fitness:\n");
    for (size_t i = 0; i < length; i++) {
        printf("%d\n", fitness[i]);
    }

    for (size_t k = 0; k < populationSize; k++) {

        memset(fitness, 0, length*sizeof(int));

        evaluation(population, itemWeight, itemProfit, fitness, length, maxWeight);
        
        /*
        for (size_t i = 0; i < length; i++) {
            printf("%d\n", fitness[i]);
        }
        */

        int parents[2][length];
        memset(parents, 0, 2*length*sizeof(int));

        selection(fitness, parents, population, length);

        /*
        printf("Parents:\n");

        for (size_t i = 0; i < 2; i++) {
            printf("[ ");
            for (size_t j = 0; j < length; j++) {
                printf("%d ", parents[i][j]);
            }
            printf("]\n");
        }
        */

        int offspring[2][length];
        memset(offspring, 0, 2*length*sizeof(int));

        crossover(parents, offspring, length);

        /*
        printf("Offsprings:\n");

        for (size_t i = 0; i < 2; i++) {
            printf("[ ");
            for (size_t j = 0; j < length; j++) {
                printf("%d ", offspring[i][j]);
            }
            printf("]\n");
        }
        */

        int mutants[2][length];
        memset(mutants, 0, 2*length*sizeof(int));

        mutation(offspring, mutants, length);

        /*
        printf("Mutants:\n");

        for (size_t i = 0; i < 2; i++) {
            printf("[ ");
            for (size_t j = 0; j < length; j++) {
                printf("%d ", mutants[i][j]);
            }
            printf("]\n");
        }
        */

        replacement(population, mutants, parents, fitness, itemWeight, itemProfit, length, maxWeight);
        
    }

    printf("New Population:\n");

    for (int i = 0; i < solutionsPerPopulation; i++) {
        printf("[ ");
        for (int j = 0; j < length; j++) {  
            printf("%d ", population[i][j]);    
        }  
        printf("]\n");
    }  
    
    evaluation(population, itemWeight, itemProfit, fitness, length, maxWeight);
    
    printf("New Fitness:\n");
    for (size_t i = 0; i < length; i++) {
        printf("%d\n", fitness[i]);
    }

    return 0;    
}