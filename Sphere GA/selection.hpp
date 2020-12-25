#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <vector>


 #define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })



void tournamentSelection(std::vector<std::vector<double>> &parents, std::vector<std::vector<double>> population, 
                         std::vector<double> fitness, const int p, const int populationSize, const int mating, const int tournamentSize) {
    
    std::vector<int> individualPool(tournamentSize, 0);

    int count = 0;
    while (count < mating) {
        
        // selecting individuals for tournament
        for (std::vector<int>::size_type i = 0; i != tournamentSize; i++) { individualPool[i] = rand() % populationSize; }

        int parentIndex = fitness[individualPool[0]] < fitness[individualPool[1]] ? individualPool[0] : individualPool[1];
        
        for (std::vector<int>::size_type i = 0; i != p; i++) { parents[count][i] = population[parentIndex][i]; }

        count++;
    }


}