#include <stdio.h> 
#include <stdlib.h>
#include <vector>

using std::vector;

void crossover (vector<vector<double>> parents, vector<vector<double>> &offsprings, const int p) {

    int crossoverPoint = rand() % p;

    for (vector<int>::size_type i = 0; i != crossoverPoint; i++) {
        offsprings[0][i] = parents[0][i];
        offsprings[1][i] = parents[1][i];
    }

    for (vector<int>::size_type i = crossoverPoint; i != p; i++) {
        offsprings[0][i] = parents[1][i];
        offsprings[1][i] = parents[0][i];
    }

}