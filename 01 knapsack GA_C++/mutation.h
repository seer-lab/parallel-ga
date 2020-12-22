#include <stdio.h> 
#include <stdlib.h>
#include <vector>

using std::vector;


void mutation (vector<vector<int>> offsprings, 
               vector<vector<int>> &mutants, const int length) {

    const float mutationRate = 0.5f;

    for(vector<int>::size_type i = 0; i != 2; i++) {
        for (vector<int>::size_type j = 0; j != length; j++) {
            mutants[i][j] = offsprings[i][j];
        }
    }

    for(vector<int>::size_type i = 0; i != 2; i++) {
        float randomVal = ((float)rand())/RAND_MAX;
        if (randomVal <= mutationRate) {
            for (vector<int>::size_type j = 0; j != length; j++)
                mutants[i][j] = ((mutants[i][j] == 0) ? 1 : 0);
        }
    }
}