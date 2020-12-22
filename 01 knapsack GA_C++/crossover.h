#include <stdio.h> 
#include <stdlib.h>
#include <vector>

using std::vector;

void crossover (vector<vector<int>> parents, vector<vector<int>> &offsprings, const int length) {

    int crossoverPoint = rand() % length;

    for (vector<int>::size_type i = 0; i != crossoverPoint; i++) {
        offsprings[0][i] = parents[0][i];
        offsprings[1][i] = parents[1][i];
    }

    for (vector<int>::size_type i = crossoverPoint; i != length; i++) {
        offsprings[0][i] = parents[1][i];
        offsprings[1][i] = parents[0][i];
    }

}