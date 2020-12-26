#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <vector>

using std::vector;


 #define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

void selection (vector<vector<double>> &parents, vector<double> fitness,
                vector<vector<double>> population, const int populationSize, const int p) {

    // generate 4 indexs between 0-9
    vector<int> individualPool(4, 0);
    for (vector<int>::size_type i = 0; i != individualPool.size(); i++) { individualPool[i] = rand() % populationSize; }
    

    int parentIndex1 = 0;
    int parentIndex2 = 0;
    
    for (vector<int>::size_type i = 0; i != fitness.size(); i++) {
        
        if (fitness[i] == min(fitness[individualPool[0]], fitness[individualPool[1]]))
            parentIndex1 = i;

        if (fitness[i] == min(fitness[individualPool[2]], fitness[individualPool[3]]))
            parentIndex2 = i;
    }

    for (vector<int>::size_type i = 0; i != p; i++) {
        parents[0][i] = population[parentIndex1][i];
        parents[1][i] = population[parentIndex2][i];
    }
}