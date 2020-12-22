#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <vector>

using std::vector;


 #define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

void selection (vector<int> fitness, vector<vector<int>> &parents, 
                vector<vector<int>> population, const int length) {

    // generate 4 indexs between 0-9
    vector<int> individualPool(4, 0);
    for (vector<int>::size_type i = 0; i != individualPool.size(); i++) { individualPool[i] = rand() % length; }

    int parentIndex1 = 0;
    int parentIndex2 = 0;
    
    for (vector<int>::size_type i = 0; i != fitness.size(); i++) {
        
        if (fitness[i] == max(fitness[individualPool[0]], fitness[individualPool[1]]))
            parentIndex1 = i;

        if (fitness[i] == max(fitness[individualPool[2]], fitness[individualPool[3]]))
            parentIndex2 = i;
    }
    
    
    for (vector<int>::size_type i = 0; i != length; i++) {
        parents[0][i] = population[parentIndex1][i];
        parents[1][i] = population[parentIndex2][i];
    }
}