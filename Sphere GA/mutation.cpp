#include "mutation.h"

void mutation (vector<vector<double>> offsprings, std::vector<float> bounds,
               vector<vector<double>> &mutants, const int p) {

    const float mutationRate = 0.05;

    for(vector<int>::size_type i = 0; i != 2; i++) {
        for (vector<int>::size_type j = 0; j != p; j++) {
            mutants[i][j] = offsprings[i][j];
        }
    }

    for(vector<int>::size_type i = 0; i != 2; i++) {
        float randomVal = ((float)rand())/RAND_MAX;
        int numGenes = rand() % p+1;
        if (randomVal <= mutationRate) {
            for (vector<int>::size_type j = 0; j != numGenes; j++)
                mutants[i][j] = bounds[0] + ((double)rand() / RAND_MAX) * (bounds[1] - bounds[0]);
        }
    }
}